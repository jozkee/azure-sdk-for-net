// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.NewRelicObservability.Models
{
    public partial class MarketplaceSaaSInfo : IUtf8JsonSerializable, IJsonModel<MarketplaceSaaSInfo>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<MarketplaceSaaSInfo>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<MarketplaceSaaSInfo>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MarketplaceSaaSInfo>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(MarketplaceSaaSInfo)} does not support writing '{format}' format.");
            }

            if (Optional.IsDefined(MarketplaceSubscriptionId))
            {
                writer.WritePropertyName("marketplaceSubscriptionId"u8);
                writer.WriteStringValue(MarketplaceSubscriptionId);
            }
            if (Optional.IsDefined(MarketplaceSubscriptionName))
            {
                writer.WritePropertyName("marketplaceSubscriptionName"u8);
                writer.WriteStringValue(MarketplaceSubscriptionName);
            }
            if (Optional.IsDefined(MarketplaceResourceId))
            {
                writer.WritePropertyName("marketplaceResourceId"u8);
                writer.WriteStringValue(MarketplaceResourceId);
            }
            if (Optional.IsDefined(MarketplaceStatus))
            {
                writer.WritePropertyName("marketplaceStatus"u8);
                writer.WriteStringValue(MarketplaceStatus);
            }
            if (Optional.IsDefined(BilledAzureSubscriptionId))
            {
                writer.WritePropertyName("billedAzureSubscriptionId"u8);
                writer.WriteStringValue(BilledAzureSubscriptionId);
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

        MarketplaceSaaSInfo IJsonModel<MarketplaceSaaSInfo>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MarketplaceSaaSInfo>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(MarketplaceSaaSInfo)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeMarketplaceSaaSInfo(document.RootElement, options);
        }

        internal static MarketplaceSaaSInfo DeserializeMarketplaceSaaSInfo(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            string marketplaceSubscriptionId = default;
            string marketplaceSubscriptionName = default;
            string marketplaceResourceId = default;
            string marketplaceStatus = default;
            string billedAzureSubscriptionId = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("marketplaceSubscriptionId"u8))
                {
                    marketplaceSubscriptionId = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("marketplaceSubscriptionName"u8))
                {
                    marketplaceSubscriptionName = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("marketplaceResourceId"u8))
                {
                    marketplaceResourceId = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("marketplaceStatus"u8))
                {
                    marketplaceStatus = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("billedAzureSubscriptionId"u8))
                {
                    billedAzureSubscriptionId = property.Value.GetString();
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new MarketplaceSaaSInfo(
                marketplaceSubscriptionId,
                marketplaceSubscriptionName,
                marketplaceResourceId,
                marketplaceStatus,
                billedAzureSubscriptionId,
                serializedAdditionalRawData);
        }

        private BinaryData SerializeBicep(ModelReaderWriterOptions options)
        {
            StringBuilder builder = new StringBuilder();
            BicepModelReaderWriterOptions bicepOptions = options as BicepModelReaderWriterOptions;
            IDictionary<string, string> propertyOverrides = null;
            bool hasObjectOverride = bicepOptions != null && bicepOptions.PropertyOverrides.TryGetValue(this, out propertyOverrides);
            bool hasPropertyOverride = false;
            string propertyOverride = null;

            builder.AppendLine("{");

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(MarketplaceSubscriptionId), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  marketplaceSubscriptionId: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(MarketplaceSubscriptionId))
                {
                    builder.Append("  marketplaceSubscriptionId: ");
                    if (MarketplaceSubscriptionId.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{MarketplaceSubscriptionId}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{MarketplaceSubscriptionId}'");
                    }
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(MarketplaceSubscriptionName), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  marketplaceSubscriptionName: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(MarketplaceSubscriptionName))
                {
                    builder.Append("  marketplaceSubscriptionName: ");
                    if (MarketplaceSubscriptionName.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{MarketplaceSubscriptionName}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{MarketplaceSubscriptionName}'");
                    }
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(MarketplaceResourceId), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  marketplaceResourceId: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(MarketplaceResourceId))
                {
                    builder.Append("  marketplaceResourceId: ");
                    if (MarketplaceResourceId.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{MarketplaceResourceId}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{MarketplaceResourceId}'");
                    }
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(MarketplaceStatus), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  marketplaceStatus: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(MarketplaceStatus))
                {
                    builder.Append("  marketplaceStatus: ");
                    if (MarketplaceStatus.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{MarketplaceStatus}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{MarketplaceStatus}'");
                    }
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(BilledAzureSubscriptionId), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  billedAzureSubscriptionId: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(BilledAzureSubscriptionId))
                {
                    builder.Append("  billedAzureSubscriptionId: ");
                    if (BilledAzureSubscriptionId.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{BilledAzureSubscriptionId}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{BilledAzureSubscriptionId}'");
                    }
                }
            }

            builder.AppendLine("}");
            return BinaryData.FromString(builder.ToString());
        }

        BinaryData IPersistableModel<MarketplaceSaaSInfo>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MarketplaceSaaSInfo>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerNewRelicObservabilityContext.Default);
                case "bicep":
                    return SerializeBicep(options);
                default:
                    throw new FormatException($"The model {nameof(MarketplaceSaaSInfo)} does not support writing '{options.Format}' format.");
            }
        }

        MarketplaceSaaSInfo IPersistableModel<MarketplaceSaaSInfo>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MarketplaceSaaSInfo>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeMarketplaceSaaSInfo(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(MarketplaceSaaSInfo)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<MarketplaceSaaSInfo>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
