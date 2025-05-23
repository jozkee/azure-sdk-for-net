// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Text.Json;

namespace Azure.ResourceManager.DataProtectionBackup
{
    public partial class DataProtectionBackupRecoveryPointResource : IJsonModel<DataProtectionBackupRecoveryPointData>
    {
        void IJsonModel<DataProtectionBackupRecoveryPointData>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options) => ((IJsonModel<DataProtectionBackupRecoveryPointData>)Data).Write(writer, options);

        DataProtectionBackupRecoveryPointData IJsonModel<DataProtectionBackupRecoveryPointData>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options) => ((IJsonModel<DataProtectionBackupRecoveryPointData>)Data).Create(ref reader, options);

        BinaryData IPersistableModel<DataProtectionBackupRecoveryPointData>.Write(ModelReaderWriterOptions options) => ModelReaderWriter.Write<DataProtectionBackupRecoveryPointData>(Data, options, AzureResourceManagerDataProtectionBackupContext.Default);

        DataProtectionBackupRecoveryPointData IPersistableModel<DataProtectionBackupRecoveryPointData>.Create(BinaryData data, ModelReaderWriterOptions options) => ModelReaderWriter.Read<DataProtectionBackupRecoveryPointData>(data, options, AzureResourceManagerDataProtectionBackupContext.Default);

        string IPersistableModel<DataProtectionBackupRecoveryPointData>.GetFormatFromOptions(ModelReaderWriterOptions options) => ((IPersistableModel<DataProtectionBackupRecoveryPointData>)Data).GetFormatFromOptions(options);
    }
}
