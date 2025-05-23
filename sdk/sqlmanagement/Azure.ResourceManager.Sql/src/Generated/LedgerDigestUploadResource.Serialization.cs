// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.ClientModel.Primitives;
using System.Text.Json;

namespace Azure.ResourceManager.Sql
{
    public partial class LedgerDigestUploadResource : IJsonModel<LedgerDigestUploadData>
    {
        void IJsonModel<LedgerDigestUploadData>.Write(Utf8JsonWriter writer, ModelReaderWriterOptions options) => ((IJsonModel<LedgerDigestUploadData>)Data).Write(writer, options);

        LedgerDigestUploadData IJsonModel<LedgerDigestUploadData>.Create(ref Utf8JsonReader reader, ModelReaderWriterOptions options) => ((IJsonModel<LedgerDigestUploadData>)Data).Create(ref reader, options);

        BinaryData IPersistableModel<LedgerDigestUploadData>.Write(ModelReaderWriterOptions options) => ModelReaderWriter.Write<LedgerDigestUploadData>(Data, options, AzureResourceManagerSqlContext.Default);

        LedgerDigestUploadData IPersistableModel<LedgerDigestUploadData>.Create(BinaryData data, ModelReaderWriterOptions options) => ModelReaderWriter.Read<LedgerDigestUploadData>(data, options, AzureResourceManagerSqlContext.Default);

        string IPersistableModel<LedgerDigestUploadData>.GetFormatFromOptions(ModelReaderWriterOptions options) => ((IPersistableModel<LedgerDigestUploadData>)Data).GetFormatFromOptions(options);
    }
}
