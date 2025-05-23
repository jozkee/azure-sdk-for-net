// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.ContainerServiceFleet.Models
{
    public partial class FleetHubProfile : IUtf8JsonSerializable, IJsonModel<FleetHubProfile>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<FleetHubProfile>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<FleetHubProfile>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<FleetHubProfile>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(FleetHubProfile)} does not support writing '{format}' format.");
            }

            if (Optional.IsDefined(DnsPrefix))
            {
                writer.WritePropertyName("dnsPrefix"u8);
                writer.WriteStringValue(DnsPrefix);
            }
            if (Optional.IsDefined(ApiServerAccessProfile))
            {
                writer.WritePropertyName("apiServerAccessProfile"u8);
                writer.WriteObjectValue(ApiServerAccessProfile, options);
            }
            if (Optional.IsDefined(AgentProfile))
            {
                writer.WritePropertyName("agentProfile"u8);
                writer.WriteObjectValue(AgentProfile, options);
            }
            if (options.Format != "W" && Optional.IsDefined(Fqdn))
            {
                writer.WritePropertyName("fqdn"u8);
                writer.WriteStringValue(Fqdn);
            }
            if (options.Format != "W" && Optional.IsDefined(KubernetesVersion))
            {
                writer.WritePropertyName("kubernetesVersion"u8);
                writer.WriteStringValue(KubernetesVersion);
            }
            if (options.Format != "W" && Optional.IsDefined(PortalFqdn))
            {
                writer.WritePropertyName("portalFqdn"u8);
                writer.WriteStringValue(PortalFqdn);
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

        FleetHubProfile IJsonModel<FleetHubProfile>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<FleetHubProfile>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(FleetHubProfile)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeFleetHubProfile(document.RootElement, options);
        }

        internal static FleetHubProfile DeserializeFleetHubProfile(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            string dnsPrefix = default;
            ContainerServiceFleetAPIServerAccessProfile apiServerAccessProfile = default;
            ContainerServiceFleetAgentProfile agentProfile = default;
            string fqdn = default;
            string kubernetesVersion = default;
            string portalFqdn = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("dnsPrefix"u8))
                {
                    dnsPrefix = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("apiServerAccessProfile"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    apiServerAccessProfile = ContainerServiceFleetAPIServerAccessProfile.DeserializeContainerServiceFleetAPIServerAccessProfile(property.Value, options);
                    continue;
                }
                if (property.NameEquals("agentProfile"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    agentProfile = ContainerServiceFleetAgentProfile.DeserializeContainerServiceFleetAgentProfile(property.Value, options);
                    continue;
                }
                if (property.NameEquals("fqdn"u8))
                {
                    fqdn = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("kubernetesVersion"u8))
                {
                    kubernetesVersion = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("portalFqdn"u8))
                {
                    portalFqdn = property.Value.GetString();
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new FleetHubProfile(
                dnsPrefix,
                apiServerAccessProfile,
                agentProfile,
                fqdn,
                kubernetesVersion,
                portalFqdn,
                serializedAdditionalRawData);
        }

        BinaryData IPersistableModel<FleetHubProfile>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<FleetHubProfile>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerContainerServiceFleetContext.Default);
                default:
                    throw new FormatException($"The model {nameof(FleetHubProfile)} does not support writing '{options.Format}' format.");
            }
        }

        FleetHubProfile IPersistableModel<FleetHubProfile>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<FleetHubProfile>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeFleetHubProfile(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(FleetHubProfile)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<FleetHubProfile>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
