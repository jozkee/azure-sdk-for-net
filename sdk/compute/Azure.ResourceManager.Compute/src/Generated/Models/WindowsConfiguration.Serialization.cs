// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Collections.Generic;
using System.Text.Json;
using Azure.Core;

namespace Azure.ResourceManager.Compute.Models
{
    public partial class WindowsConfiguration : IUtf8JsonSerializable, IJsonModel<WindowsConfiguration>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<WindowsConfiguration>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<WindowsConfiguration>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected virtual void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<WindowsConfiguration>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(WindowsConfiguration)} does not support writing '{format}' format.");
            }

            if (Optional.IsDefined(ProvisionVmAgent))
            {
                writer.WritePropertyName("provisionVMAgent"u8);
                writer.WriteBooleanValue(ProvisionVmAgent.Value);
            }
            if (Optional.IsDefined(IsAutomaticUpdatesEnabled))
            {
                writer.WritePropertyName("enableAutomaticUpdates"u8);
                writer.WriteBooleanValue(IsAutomaticUpdatesEnabled.Value);
            }
            if (Optional.IsDefined(TimeZone))
            {
                writer.WritePropertyName("timeZone"u8);
                writer.WriteStringValue(TimeZone);
            }
            if (Optional.IsCollectionDefined(AdditionalUnattendContent))
            {
                writer.WritePropertyName("additionalUnattendContent"u8);
                writer.WriteStartArray();
                foreach (var item in AdditionalUnattendContent)
                {
                    writer.WriteObjectValue(item, options);
                }
                writer.WriteEndArray();
            }
            if (Optional.IsDefined(PatchSettings))
            {
                writer.WritePropertyName("patchSettings"u8);
                writer.WriteObjectValue(PatchSettings, options);
            }
            if (Optional.IsDefined(WinRM))
            {
                writer.WritePropertyName("winRM"u8);
                writer.WriteObjectValue(WinRM, options);
            }
            if (options.Format != "W" && Optional.IsDefined(IsVmAgentPlatformUpdatesEnabled))
            {
                writer.WritePropertyName("enableVMAgentPlatformUpdates"u8);
                writer.WriteBooleanValue(IsVmAgentPlatformUpdatesEnabled.Value);
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

        WindowsConfiguration IJsonModel<WindowsConfiguration>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<WindowsConfiguration>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(WindowsConfiguration)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeWindowsConfiguration(document.RootElement, options);
        }

        internal static WindowsConfiguration DeserializeWindowsConfiguration(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            bool? provisionVmAgent = default;
            bool? enableAutomaticUpdates = default;
            string timeZone = default;
            IList<AdditionalUnattendContent> additionalUnattendContent = default;
            PatchSettings patchSettings = default;
            WinRMConfiguration winRM = default;
            bool? enableVmAgentPlatformUpdates = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("provisionVMAgent"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    provisionVmAgent = property.Value.GetBoolean();
                    continue;
                }
                if (property.NameEquals("enableAutomaticUpdates"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    enableAutomaticUpdates = property.Value.GetBoolean();
                    continue;
                }
                if (property.NameEquals("timeZone"u8))
                {
                    timeZone = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("additionalUnattendContent"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    List<AdditionalUnattendContent> array = new List<AdditionalUnattendContent>();
                    foreach (var item in property.Value.EnumerateArray())
                    {
                        array.Add(Models.AdditionalUnattendContent.DeserializeAdditionalUnattendContent(item, options));
                    }
                    additionalUnattendContent = array;
                    continue;
                }
                if (property.NameEquals("patchSettings"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    patchSettings = PatchSettings.DeserializePatchSettings(property.Value, options);
                    continue;
                }
                if (property.NameEquals("winRM"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    winRM = WinRMConfiguration.DeserializeWinRMConfiguration(property.Value, options);
                    continue;
                }
                if (property.NameEquals("enableVMAgentPlatformUpdates"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    enableVmAgentPlatformUpdates = property.Value.GetBoolean();
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new WindowsConfiguration(
                provisionVmAgent,
                enableAutomaticUpdates,
                timeZone,
                additionalUnattendContent ?? new ChangeTrackingList<AdditionalUnattendContent>(),
                patchSettings,
                winRM,
                enableVmAgentPlatformUpdates,
                serializedAdditionalRawData);
        }

        BinaryData IPersistableModel<WindowsConfiguration>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<WindowsConfiguration>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerComputeContext.Default);
                default:
                    throw new FormatException($"The model {nameof(WindowsConfiguration)} does not support writing '{options.Format}' format.");
            }
        }

        WindowsConfiguration IPersistableModel<WindowsConfiguration>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<WindowsConfiguration>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeWindowsConfiguration(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(WindowsConfiguration)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<WindowsConfiguration>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
