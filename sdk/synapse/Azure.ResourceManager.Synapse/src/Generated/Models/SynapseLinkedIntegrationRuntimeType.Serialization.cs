// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.Synapse.Models
{
    [PersistableModelProxy(typeof(UnknownLinkedIntegrationRuntimeType))]
    public partial class SynapseLinkedIntegrationRuntimeType : IUtf8JsonSerializable, IJsonModel<SynapseLinkedIntegrationRuntimeType>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<SynapseLinkedIntegrationRuntimeType>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<SynapseLinkedIntegrationRuntimeType>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<SynapseLinkedIntegrationRuntimeType>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(SynapseLinkedIntegrationRuntimeType)} does not support writing '{format}' format.");
            }

            writer.WritePropertyName("authorizationType"u8);
            writer.WriteStringValue(AuthorizationType);
            if (options.Format != "W" && _serializedAdditionalRawData != null)
            {
                foreach (var item in _serializedAdditionalRawData)
                {
                    writer.WritePropertyName(item.Key);
#if NET6_0_OR_GREATER
				writer.WriteRawValue(item.Value);
#else
                    using (JsonDocument document = JsonDocument.Parse(item.Value, ModelSerializationExtensions.JsonDocumentOptions))
                    {
                        JsonSerializer.Serialize(writer, document.RootElement);
                    }
#endif
                }
            }
        }

        SynapseLinkedIntegrationRuntimeType IJsonModel<SynapseLinkedIntegrationRuntimeType>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<SynapseLinkedIntegrationRuntimeType>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(SynapseLinkedIntegrationRuntimeType)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeSynapseLinkedIntegrationRuntimeType(document.RootElement, options);
        }

        internal static SynapseLinkedIntegrationRuntimeType DeserializeSynapseLinkedIntegrationRuntimeType(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            if (element.TryGetProperty("authorizationType", out JsonElement discriminator))
            {
                switch (discriminator.GetString())
                {
                    case "Key": return SynapseLinkedIntegrationRuntimeKeyAuthorization.DeserializeSynapseLinkedIntegrationRuntimeKeyAuthorization(element, options);
                    case "RBAC": return SynapseLinkedIntegrationRuntimeRbacAuthorization.DeserializeSynapseLinkedIntegrationRuntimeRbacAuthorization(element, options);
                }
            }
            return UnknownLinkedIntegrationRuntimeType.DeserializeUnknownLinkedIntegrationRuntimeType(element, options);
        }

        BinaryData IPersistableModel<SynapseLinkedIntegrationRuntimeType>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<SynapseLinkedIntegrationRuntimeType>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerSynapseContext.Default);
                default:
                    throw new FormatException($"The model {nameof(SynapseLinkedIntegrationRuntimeType)} does not support writing '{options.Format}' format.");
            }
        }

        SynapseLinkedIntegrationRuntimeType IPersistableModel<SynapseLinkedIntegrationRuntimeType>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<SynapseLinkedIntegrationRuntimeType>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeSynapseLinkedIntegrationRuntimeType(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(SynapseLinkedIntegrationRuntimeType)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<SynapseLinkedIntegrationRuntimeType>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
