// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Microsoft.Extensions.AI;

/// <summary>Used to create the JSON payload for an AzureAI chat tool description.</summary>
internal sealed class AzureAIChatToolJson
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = "object";

    [JsonPropertyName("required")]
    public List<string> Required { get; set; } = [];

    [JsonPropertyName("properties")]
    public Dictionary<string, JsonElement> Properties { get; set; } = [];
}
