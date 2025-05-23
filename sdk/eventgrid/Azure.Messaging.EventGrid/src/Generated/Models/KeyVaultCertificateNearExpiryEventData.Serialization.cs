// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Azure.Messaging.EventGrid.SystemEvents
{
    [JsonConverter(typeof(KeyVaultCertificateNearExpiryEventDataConverter))]
    public partial class KeyVaultCertificateNearExpiryEventData
    {
        internal static KeyVaultCertificateNearExpiryEventData DeserializeKeyVaultCertificateNearExpiryEventData(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            string id = default;
            string vaultName = default;
            string objectType = default;
            string objectName = default;
            string version = default;
            float? nbf = default;
            float? exp = default;
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("Id"u8))
                {
                    id = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("VaultName"u8))
                {
                    vaultName = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("ObjectType"u8))
                {
                    objectType = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("ObjectName"u8))
                {
                    objectName = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("Version"u8))
                {
                    version = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("NBF"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    nbf = property.Value.GetSingle();
                    continue;
                }
                if (property.NameEquals("EXP"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    exp = property.Value.GetSingle();
                    continue;
                }
            }
            return new KeyVaultCertificateNearExpiryEventData(
                id,
                vaultName,
                objectType,
                objectName,
                version,
                nbf,
                exp);
        }

        /// <summary> Deserializes the model from a raw response. </summary>
        /// <param name="response"> The response to deserialize the model from. </param>
        internal static KeyVaultCertificateNearExpiryEventData FromResponse(Response response)
        {
            using var document = JsonDocument.Parse(response.Content, ModelSerializationExtensions.JsonDocumentOptions);
            return DeserializeKeyVaultCertificateNearExpiryEventData(document.RootElement);
        }

        internal partial class KeyVaultCertificateNearExpiryEventDataConverter : JsonConverter<KeyVaultCertificateNearExpiryEventData>
        {
            public override void Write(Utf8JsonWriter writer, KeyVaultCertificateNearExpiryEventData model, JsonSerializerOptions options)
            {
                throw new NotImplementedException();
            }

            public override KeyVaultCertificateNearExpiryEventData Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            {
                using var document = JsonDocument.ParseValue(ref reader);
                return DeserializeKeyVaultCertificateNearExpiryEventData(document.RootElement);
            }
        }
    }
}
