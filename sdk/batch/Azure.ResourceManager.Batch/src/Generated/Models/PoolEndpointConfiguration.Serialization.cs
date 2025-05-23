// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.Batch.Models
{
    internal partial class PoolEndpointConfiguration : IUtf8JsonSerializable, IJsonModel<PoolEndpointConfiguration>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<PoolEndpointConfiguration>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<PoolEndpointConfiguration>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<PoolEndpointConfiguration>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(PoolEndpointConfiguration)} does not support writing '{format}' format.");
            }

            writer.WritePropertyName("inboundNatPools"u8);
            writer.WriteStartArray();
            foreach (var item in InboundNatPools)
            {
                writer.WriteObjectValue(item, options);
            }
            writer.WriteEndArray();
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

        PoolEndpointConfiguration IJsonModel<PoolEndpointConfiguration>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<PoolEndpointConfiguration>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(PoolEndpointConfiguration)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializePoolEndpointConfiguration(document.RootElement, options);
        }

        internal static PoolEndpointConfiguration DeserializePoolEndpointConfiguration(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            IList<BatchInboundNatPool> inboundNatPools = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("inboundNatPools"u8))
                {
                    List<BatchInboundNatPool> array = new List<BatchInboundNatPool>();
                    foreach (var item in property.Value.EnumerateArray())
                    {
                        array.Add(BatchInboundNatPool.DeserializeBatchInboundNatPool(item, options));
                    }
                    inboundNatPools = array;
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new PoolEndpointConfiguration(inboundNatPools, serializedAdditionalRawData);
        }

        BinaryData IPersistableModel<PoolEndpointConfiguration>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<PoolEndpointConfiguration>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerBatchContext.Default);
                default:
                    throw new FormatException($"The model {nameof(PoolEndpointConfiguration)} does not support writing '{options.Format}' format.");
            }
        }

        PoolEndpointConfiguration IPersistableModel<PoolEndpointConfiguration>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<PoolEndpointConfiguration>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializePoolEndpointConfiguration(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(PoolEndpointConfiguration)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<PoolEndpointConfiguration>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
