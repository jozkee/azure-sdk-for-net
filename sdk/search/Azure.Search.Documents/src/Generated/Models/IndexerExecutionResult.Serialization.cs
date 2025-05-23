// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.Collections.Generic;
using System.Text.Json;

namespace Azure.Search.Documents.Indexes.Models
{
    public partial class IndexerExecutionResult
    {
        internal static IndexerExecutionResult DeserializeIndexerExecutionResult(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Null)
            {
                return null;
            }
            IndexerExecutionStatus status = default;
            IndexerExecutionStatusDetail? statusDetail = default;
            IndexingMode? mode = default;
            string errorMessage = default;
            DateTimeOffset? startTime = default;
            DateTimeOffset? endTime = default;
            IReadOnlyList<SearchIndexerError> errors = default;
            IReadOnlyList<SearchIndexerWarning> warnings = default;
            int itemsProcessed = default;
            int itemsFailed = default;
            string initialTrackingState = default;
            string finalTrackingState = default;
            foreach (var property in element.EnumerateObject())
            {
                if (property.NameEquals("status"u8))
                {
                    status = property.Value.GetString().ToIndexerExecutionStatus();
                    continue;
                }
                if (property.NameEquals("statusDetail"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        statusDetail = null;
                        continue;
                    }
                    statusDetail = new IndexerExecutionStatusDetail(property.Value.GetString());
                    continue;
                }
                if (property.NameEquals("mode"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    mode = new IndexingMode(property.Value.GetString());
                    continue;
                }
                if (property.NameEquals("errorMessage"u8))
                {
                    errorMessage = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("startTime"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        continue;
                    }
                    startTime = property.Value.GetDateTimeOffset("O");
                    continue;
                }
                if (property.NameEquals("endTime"u8))
                {
                    if (property.Value.ValueKind == JsonValueKind.Null)
                    {
                        endTime = null;
                        continue;
                    }
                    endTime = property.Value.GetDateTimeOffset("O");
                    continue;
                }
                if (property.NameEquals("errors"u8))
                {
                    List<SearchIndexerError> array = new List<SearchIndexerError>();
                    foreach (var item in property.Value.EnumerateArray())
                    {
                        array.Add(SearchIndexerError.DeserializeSearchIndexerError(item));
                    }
                    errors = array;
                    continue;
                }
                if (property.NameEquals("warnings"u8))
                {
                    List<SearchIndexerWarning> array = new List<SearchIndexerWarning>();
                    foreach (var item in property.Value.EnumerateArray())
                    {
                        array.Add(SearchIndexerWarning.DeserializeSearchIndexerWarning(item));
                    }
                    warnings = array;
                    continue;
                }
                if (property.NameEquals("itemsProcessed"u8))
                {
                    itemsProcessed = property.Value.GetInt32();
                    continue;
                }
                if (property.NameEquals("itemsFailed"u8))
                {
                    itemsFailed = property.Value.GetInt32();
                    continue;
                }
                if (property.NameEquals("initialTrackingState"u8))
                {
                    initialTrackingState = property.Value.GetString();
                    continue;
                }
                if (property.NameEquals("finalTrackingState"u8))
                {
                    finalTrackingState = property.Value.GetString();
                    continue;
                }
            }
            return new IndexerExecutionResult(
                status,
                statusDetail,
                mode,
                errorMessage,
                startTime,
                endTime,
                errors,
                warnings,
                itemsProcessed,
                itemsFailed,
                initialTrackingState,
                finalTrackingState);
        }

        /// <summary> Deserializes the model from a raw response. </summary>
        /// <param name="response"> The response to deserialize the model from. </param>
        internal static IndexerExecutionResult FromResponse(Response response)
        {
            using var document = JsonDocument.Parse(response.Content, ModelSerializationExtensions.JsonDocumentOptions);
            return DeserializeIndexerExecutionResult(document.RootElement);
        }
    }
}
