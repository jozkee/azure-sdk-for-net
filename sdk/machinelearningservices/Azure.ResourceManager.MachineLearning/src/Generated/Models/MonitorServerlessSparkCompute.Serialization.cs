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

namespace Azure.ResourceManager.MachineLearning.Models
{
    public partial class MonitorServerlessSparkCompute : IUtf8JsonSerializable, IJsonModel<MonitorServerlessSparkCompute>
    {
        void IUtf8JsonSerializable.Write(Utf8JsonWriter writer) => ((IJsonModel<MonitorServerlessSparkCompute>)this).Write(writer, ModelSerializationExtensions.WireOptions);

        void IJsonModel<MonitorServerlessSparkCompute>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            writer.WriteStartObject();
            JsonModelWriteCore(writer, options);
            writer.WriteEndObject();
        }

        /// <param name="writer"> The JSON writer. </param>
        /// <param name="options"> The client options for reading and writing models. </param>
        protected override void JsonModelWriteCore(Utf8JsonWriter writer, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MonitorServerlessSparkCompute>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(MonitorServerlessSparkCompute)} does not support writing '{format}' format.");
            }

            base.JsonModelWriteCore(writer, options);
            writer.WritePropertyName("computeIdentity"u8);
            writer.WriteObjectValue(ComputeIdentity, options);
            writer.WritePropertyName("instanceType"u8);
            writer.WriteStringValue(InstanceType);
            writer.WritePropertyName("runtimeVersion"u8);
            writer.WriteStringValue(RuntimeVersion);
        }

        MonitorServerlessSparkCompute IJsonModel<MonitorServerlessSparkCompute>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MonitorServerlessSparkCompute>)this).GetFormatFromOptions(options) : options.Format;
            if (format != "J")
            {
                throw new FormatException($"The model {nameof(MonitorServerlessSparkCompute)} does not support reading '{format}' format.");
            }

            using JsonDocument document = JsonDocument.ParseValue(ref reader);
            return DeserializeMonitorServerlessSparkCompute(document.RootElement, options);
        }

        internal static MonitorServerlessSparkCompute DeserializeMonitorServerlessSparkCompute(JsonElement element, ModelReaderWriterOptions options = null)
        {
            options ??= ModelSerializationExtensions.WireOptions;

            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            MonitorComputeIdentityBase computeIdentity = default;
            string instanceType = default;
            string runtimeVersion = default;
            MonitorComputeType computeType = default;
            IDictionary<string, BinaryData> serializedAdditionalRawData = default;
            Dictionary<string, BinaryData> rawDataDictionary = new Dictionary<string, BinaryData>();
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("computeIdentity"u8))
                {
                    computeIdentity = MonitorComputeIdentityBase.DeserializeMonitorComputeIdentityBase(property.Value, options);
                    continue;
                }
                if (property.NameEquals("instanceType"u8))
                {
                    instanceType = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("runtimeVersion"u8))
                {
                    runtimeVersion = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("computeType"u8))
                {
                    computeType = new MonitorComputeType(property.Value.GetString());
                    continue;
                }
                if (options.Format != "W")
                {
                    rawDataDictionary.Add(property.Name, BinaryData.FromString(property.Value.GetRawText()));
                }
            }
            serializedAdditionalRawData = rawDataDictionary;
            return new MonitorServerlessSparkCompute(computeType, serializedAdditionalRawData, computeIdentity, instanceType, runtimeVersion);
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

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(ComputeIdentity), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  computeIdentity: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(ComputeIdentity))
                {
                    builder.Append("  computeIdentity: ");
                    BicepSerializationHelpers.AppendChildObject(builder, ComputeIdentity, options, 2, false, "  computeIdentity: ");
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(InstanceType), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  instanceType: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(InstanceType))
                {
                    builder.Append("  instanceType: ");
                    if (InstanceType.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{InstanceType}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{InstanceType}'");
                    }
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(RuntimeVersion), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  runtimeVersion: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                if (Optional.IsDefined(RuntimeVersion))
                {
                    builder.Append("  runtimeVersion: ");
                    if (RuntimeVersion.Contains(Environment.NewLine))
                    {
                        builder.AppendLine("'''");
                        builder.AppendLine($"{RuntimeVersion}'''");
                    }
                    else
                    {
                        builder.AppendLine($"'{RuntimeVersion}'");
                    }
                }
            }

            hasPropertyOverride = hasObjectOverride && propertyOverrides.TryGetValue(nameof(ComputeType), out propertyOverride);
            if (hasPropertyOverride)
            {
                builder.Append("  computeType: ");
                builder.AppendLine(propertyOverride);
            }
            else
            {
                builder.Append("  computeType: ");
                builder.AppendLine($"'{ComputeType.ToString()}'");
            }

            builder.AppendLine("}");
            return BinaryData.FromString(builder.ToString());
        }

        BinaryData IPersistableModel<MonitorServerlessSparkCompute>.Write(ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MonitorServerlessSparkCompute>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    return ModelReaderWriter.Write(this, options, AzureResourceManagerMachineLearningContext.Default);
                case "bicep":
                    return SerializeBicep(options);
                default:
                    throw new FormatException($"The model {nameof(MonitorServerlessSparkCompute)} does not support writing '{options.Format}' format.");
            }
        }

        MonitorServerlessSparkCompute IPersistableModel<MonitorServerlessSparkCompute>.Create(BinaryData data, ModelReaderWriterOptions options)
        {
            var format = options.Format == "W" ? ((IPersistableModel<MonitorServerlessSparkCompute>)this).GetFormatFromOptions(options) : options.Format;

            switch (format)
            {
                case "J":
                    {
                        using JsonDocument document = JsonDocument.Parse(data, ModelSerializationExtensions.JsonDocumentOptions);
                        return DeserializeMonitorServerlessSparkCompute(document.RootElement, options);
                    }
                default:
                    throw new FormatException($"The model {nameof(MonitorServerlessSparkCompute)} does not support reading '{options.Format}' format.");
            }
        }

        string IPersistableModel<MonitorServerlessSparkCompute>.GetFormatFromOptions(ModelReaderWriterOptions options) => "J";
    }
}
