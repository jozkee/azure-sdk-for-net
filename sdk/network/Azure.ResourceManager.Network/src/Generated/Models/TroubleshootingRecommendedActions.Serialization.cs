// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.Network.Models
{
    public partial class TroubleshootingRecommendedActions : IUtf8JsonSerializable, IJsonModel<TroubleshootingRecommendedActions>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<TroubleshootingRecommendedActions>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<TroubleshootingRecommendedActions>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<TroubleshootingRecommendedActions>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(TroubleshootingRecommendedActions)} does not support writing '{format}' format.");
            }

            if (Optional.IsDefined(ActionId))
            {
                writer.WritePropertyName("actionId"u8);
                writer.WriteStringValue(ActionId);
            }
            if (Optional.IsDefined(ActionText))
            {
                writer.WritePropertyName("actionText"u8);
                writer.WriteStringValue(ActionText);
            }
            if (Optional.IsDefined(ActionUri))
            {
                writer.WritePropertyName("actionUri"u8);
                writer.WriteStringValue(ActionUri.AbsoluteUri);
            }
            if (Optional.IsDefined(ActionUriText))
            {
                writer.WritePropertyName("actionUriText"u8);
                writer.WriteStringValue(ActionUriText);
            }
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

        TroubleshootingRecommendedActions IJsonModel<TroubleshootingRecommendedActions>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<TroubleshootingRecommendedActions>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(TroubleshootingRecommendedActions)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeTroubleshootingRecommendedActions(document.RootElement, options);
        }

        internal static TroubleshootingRecommendedActions DeserializeTroubleshootingRecommendedActions(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            string actionId = default;
            string actionText = default;
            Uri actionUri = default;
            string actionUriText = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("actionId"u8))
                {
                    actionId = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("actionText"u8))
                {
                    actionText = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("actionUri"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    actionUri = new Uri(property.Value.GetString());
                    continue;
                }
                if (property.NameEquals("actionUriText"u8))
                {
                    actionUriText = property.Value.GetString();
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new TroubleshootingRecommendedActions(actionId, actionText, actionUri, actionUriText, serializedAdditionalRawData);
        }

        BinaryData IPersistableModel<TroubleshootingRecommendedActions>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<TroubleshootingRecommendedActions>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerNetworkContext.Default);
                default:
                    throw new FormatException($"The model {nameof(TroubleshootingRecommendedActions)} does not support writing '{options.Format}' format.");
            }
        }

        TroubleshootingRecommendedActions IPersistableModel<TroubleshootingRecommendedActions>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<TroubleshootingRecommendedActions>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeTroubleshootingRecommendedActions(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(TroubleshootingRecommendedActions)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<TroubleshootingRecommendedActions>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
