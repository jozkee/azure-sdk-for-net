// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#nullable enable

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Buffers.Text;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Azure.AI.Inference;

namespace Microsoft.Extensions.AI;

/// <summary>Represents an <see cref="IEmbeddingGenerator{String, Embedding}"/> for an Azure.AI.Inference <see cref="EmbeddingsClient"/>.</summary>
internal sealed class AzureAIInferenceEmbeddingGenerator :
    IEmbeddingGenerator<string, Embedding<float>>
{
    /// <summary>Metadata about the embedding generator.</summary>
    private readonly EmbeddingGeneratorMetadata _metadata;

    /// <summary>The underlying <see cref="EmbeddingsClient" />.</summary>
    private readonly EmbeddingsClient _embeddingsClient;

    /// <summary>The number of dimensions produced by the generator.</summary>
    private readonly int? _dimensions;

    /// <summary>Initializes a new instance of the <see cref="AzureAIInferenceEmbeddingGenerator"/> class.</summary>
    /// <param name="embeddingsClient">The underlying client.</param>
    /// <param name="defaultModelId">
    /// The ID of the model to use. This can also be overridden per request via <see cref="EmbeddingGenerationOptions.ModelId"/>.
    /// Either this parameter or <see cref="EmbeddingGenerationOptions.ModelId"/> must provide a valid model ID.
    /// </param>
    /// <param name="defaultModelDimensions">The number of dimensions to generate in each embedding.</param>
    /// <exception cref="ArgumentNullException"><paramref name="embeddingsClient"/> is <see langword="null"/>.</exception>
    /// <exception cref="ArgumentException"><paramref name="defaultModelId"/> is empty or composed entirely of whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="defaultModelDimensions"/> is not positive.</exception>
    public AzureAIInferenceEmbeddingGenerator(
        EmbeddingsClient embeddingsClient, string? defaultModelId = null, int? defaultModelDimensions = null)
    {
        if (embeddingsClient is null)
        {
            throw new ArgumentNullException(nameof(embeddingsClient));
        }

        if (defaultModelId is not null && string.IsNullOrWhiteSpace(defaultModelId))
        {
            throw new ArgumentException(nameof(defaultModelId), "Argument is whitespace");
        }

        if (defaultModelDimensions is < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(defaultModelDimensions), "Value must be greater than 0.");
        }

        _embeddingsClient = embeddingsClient;
        _dimensions = defaultModelDimensions;

        _metadata = new EmbeddingGeneratorMetadata("az.ai.inference", embeddingsClient.Endpoint, defaultModelId, defaultModelDimensions);
    }

    /// <inheritdoc />
    object? IEmbeddingGenerator.GetService(Type serviceType, object? serviceKey)
    {
        if (serviceType is null)
        {
            throw new ArgumentNullException(nameof(serviceType));
        }

        return
            serviceKey is not null ? null :
            serviceType == typeof(EmbeddingsClient) ? _embeddingsClient :
            serviceType == typeof(EmbeddingGeneratorMetadata) ? _metadata :
            serviceType.IsInstanceOfType(this) ? this :
            null;
    }

    /// <inheritdoc />
    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
    {
        var azureAIOptions = ToAzureAIOptions(values, options, EmbeddingEncodingFormat.Base64);

        var embeddings = (await _embeddingsClient.EmbedAsync(azureAIOptions, cancellationToken).ConfigureAwait(false)).Value;

        GeneratedEmbeddings<Embedding<float>> result = new(embeddings.Data.Select(e =>
            new Embedding<float>(ParseBase64Floats(e.Embedding))
            {
                CreatedAt = DateTimeOffset.UtcNow,
                ModelId = embeddings.Model ?? azureAIOptions.Model,
            }));

        if (embeddings.Usage is not null)
        {
            result.Usage = new()
            {
                InputTokenCount = embeddings.Usage.PromptTokens,
                TotalTokenCount = embeddings.Usage.TotalTokens
            };
        }

        return result;
    }

    /// <inheritdoc />
    void IDisposable.Dispose()
    {
        // Nothing to dispose. Implementation required for the IEmbeddingGenerator interface.
    }

    private static float[] ParseBase64Floats(BinaryData binaryData)
    {
        ReadOnlySpan<byte> base64 = binaryData.ToMemory().Span;

        // Remove quotes around base64 string.
        if (base64.Length < 2 || base64[0] != (byte)'"' || base64[base64.Length - 1] != (byte)'"')
        {
            ThrowInvalidData();
        }

        base64 = base64.Slice(1, base64.Length - 2);

        // Decode base64 string to bytes.
        byte[] bytes = ArrayPool<byte>.Shared.Rent(Base64.GetMaxDecodedFromUtf8Length(base64.Length));
        OperationStatus status = Base64.DecodeFromUtf8(base64, bytes.AsSpan(), out int bytesConsumed, out int bytesWritten);
        if (status != OperationStatus.Done || bytesWritten % sizeof(float) != 0)
        {
            ThrowInvalidData();
        }

        // Interpret bytes as floats
        float[] vector = new float[bytesWritten / sizeof(float)];
        bytes.AsSpan(0, bytesWritten).CopyTo(MemoryMarshal.AsBytes(vector.AsSpan()));
        if (!BitConverter.IsLittleEndian)
        {
            Span<int> ints = MemoryMarshal.Cast<float, int>(vector.AsSpan());
#if NET
            BinaryPrimitives.ReverseEndianness(ints, ints);
#else
            for (int i = 0; i < ints.Length; i++)
            {
                ints[i] = BinaryPrimitives.ReverseEndianness(ints[i]);
            }
#endif
        }

        ArrayPool<byte>.Shared.Return(bytes);
        return vector;

        static void ThrowInvalidData() =>
            throw new FormatException("The input is not a valid Base64 string of encoded floats.");
    }

    /// <summary>Converts an extensions options instance to an OpenAI options instance.</summary>
    private EmbeddingsOptions ToAzureAIOptions(IEnumerable<string> inputs, EmbeddingGenerationOptions? options, EmbeddingEncodingFormat format)
    {
        EmbeddingsOptions result = new(inputs)
        {
            Dimensions = options?.Dimensions ?? _dimensions,
            Model = options?.ModelId ?? _metadata.DefaultModelId,
            EncodingFormat = format,
        };

        if (options?.AdditionalProperties is { } props)
        {
            foreach (var prop in props)
            {
                if (prop.Value is not null)
                {
                    byte[] data = JsonSerializer.SerializeToUtf8Bytes(prop.Value, AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(object)));
                    result.AdditionalProperties[prop.Key] = new BinaryData(data);
                }
            }
        }

        return result;
    }
}
