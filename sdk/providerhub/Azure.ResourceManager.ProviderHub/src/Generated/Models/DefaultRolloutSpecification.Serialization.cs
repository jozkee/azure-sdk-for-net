// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.ProviderHub.Models
{
    public partial class DefaultRolloutSpecification : IUtf8JsonSerializable, IJsonModel<DefaultRolloutSpecification>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<DefaultRolloutSpecification>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<DefaultRolloutSpecification>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<DefaultRolloutSpecification>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(DefaultRolloutSpecification)} does not support writing '{format}' format.");
            }

            if (Optional.IsDefined(Canary))
            {
                writer.WritePropertyName("canary"u8);
                writer.WriteObjectValue(Canary, options);
            }
            if (Optional.IsDefined(LowTraffic))
            {
                writer.WritePropertyName("lowTraffic"u8);
                writer.WriteObjectValue(LowTraffic, options);
            }
            if (Optional.IsDefined(MediumTraffic))
            {
                writer.WritePropertyName("mediumTraffic"u8);
                writer.WriteObjectValue(MediumTraffic, options);
            }
            if (Optional.IsDefined(HighTraffic))
            {
                writer.WritePropertyName("highTraffic"u8);
                writer.WriteObjectValue(HighTraffic, options);
            }
            if (Optional.IsDefined(RestOfTheWorldGroupOne))
            {
                writer.WritePropertyName("restOfTheWorldGroupOne"u8);
                writer.WriteObjectValue(RestOfTheWorldGroupOne, options);
            }
            if (Optional.IsDefined(RestOfTheWorldGroupTwo))
            {
                writer.WritePropertyName("restOfTheWorldGroupTwo"u8);
                writer.WriteObjectValue(RestOfTheWorldGroupTwo, options);
            }
            if (Optional.IsDefined(ProviderRegistration))
            {
                writer.WritePropertyName("providerRegistration"u8);
                writer.WriteObjectValue(ProviderRegistration, options);
            }
            if (Optional.IsCollectionDefined(ResourceTypeRegistrations))
            {
                writer.WritePropertyName("resourceTypeRegistrations"u8);
                writer.WriteStartArray();
                foreach (var item in ResourceTypeRegistrations)
                {
                    writer.WriteObjectValue(item, options);
                }
                writer.WriteEndArray();
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

        DefaultRolloutSpecification IJsonModel<DefaultRolloutSpecification>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<DefaultRolloutSpecification>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(DefaultRolloutSpecification)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeDefaultRolloutSpecification(document.RootElement, options);
        }

        internal static DefaultRolloutSpecification DeserializeDefaultRolloutSpecification(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            CanaryTrafficRegionRolloutConfiguration canary = default;
            TrafficRegionRolloutConfiguration lowTraffic = default;
            TrafficRegionRolloutConfiguration mediumTraffic = default;
            TrafficRegionRolloutConfiguration highTraffic = default;
            TrafficRegionRolloutConfiguration restOfTheWorldGroupOne = default;
            TrafficRegionRolloutConfiguration restOfTheWorldGroupTwo = default;
            ProviderRegistrationData providerRegistration = default;
            IList<ResourceTypeRegistrationData> resourceTypeRegistrations = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("canary"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    canary = CanaryTrafficRegionRolloutConfiguration.DeserializeCanaryTrafficRegionRolloutConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("lowTraffic"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    lowTraffic = TrafficRegionRolloutConfiguration.DeserializeTrafficRegionRolloutConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("mediumTraffic"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    mediumTraffic = TrafficRegionRolloutConfiguration.DeserializeTrafficRegionRolloutConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("highTraffic"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    highTraffic = TrafficRegionRolloutConfiguration.DeserializeTrafficRegionRolloutConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("restOfTheWorldGroupOne"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    restOfTheWorldGroupOne = TrafficRegionRolloutConfiguration.DeserializeTrafficRegionRolloutConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("restOfTheWorldGroupTwo"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    restOfTheWorldGroupTwo = TrafficRegionRolloutConfiguration.DeserializeTrafficRegionRolloutConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("providerRegistration"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    providerRegistration = ProviderRegistrationData.DeserializeProviderRegistrationData(property.Value, options);
                    continue;
                }
                if (property.NameEquals("resourceTypeRegistrations"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    List<ResourceTypeRegistrationData> array = new List<ResourceTypeRegistrationData>();
                    foreach (var item in property.Value.EnumerateArray())
                    {
                        array.Add(ResourceTypeRegistrationData.DeserializeResourceTypeRegistrationData(item, options));
                    }
                    resourceTypeRegistrations = array;
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new DefaultRolloutSpecification(
                canary,
                lowTraffic,
                mediumTraffic,
                highTraffic,
                restOfTheWorldGroupOne,
                restOfTheWorldGroupTwo,
                providerRegistration,
                resourceTypeRegistrations ?? new ChangeTrackingList<ResourceTypeRegistrationData>(),
                serializedAdditionalRawData);
        }

        BinaryData IPersistableModel<DefaultRolloutSpecification>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<DefaultRolloutSpecification>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerProviderHubContext.Default);
                default:
                    throw new FormatException($"The model {nameof(DefaultRolloutSpecification)} does not support writing '{options.Format}' format.");
            }
        }

        DefaultRolloutSpecification IPersistableModel<DefaultRolloutSpecification>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<DefaultRolloutSpecification>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeDefaultRolloutSpecification(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(DefaultRolloutSpecification)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<DefaultRolloutSpecification>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
