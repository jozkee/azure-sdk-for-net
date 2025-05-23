// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System.ClientModel.Primitives;
using System.Threading;
using System.Threading.Tasks;
using Azure.Core;

namespace Azure.ResourceManager.PostgreSql
{
    internal class PostgreSqlFirewallRuleOperationSource : IOperationSource<PostgreSqlFirewallRuleResource>
    {
        private readonly ArmClient _client;

        internal PostgreSqlFirewallRuleOperationSource(ArmClient client)
        {
            _client = client;
        }

        PostgreSqlFirewallRuleResource IOperationSource<PostgreSqlFirewallRuleResource>.CreateResult(Response response, CancellationToken cancellationToken)
        {
            var data = ModelReaderWriter.Read<PostgreSqlFirewallRuleData>(response.Content, ModelReaderWriterOptions.Json, AzureResourceManagerPostgreSqlContext.Default);
            return new PostgreSqlFirewallRuleResource(_client, data);
        }

        async ValueTask<PostgreSqlFirewallRuleResource> IOperationSource<PostgreSqlFirewallRuleResource>.CreateResultAsync(Response response, CancellationToken cancellationToken)
        {
            var data = ModelReaderWriter.Read<PostgreSqlFirewallRuleData>(response.Content, ModelReaderWriterOptions.Json, AzureResourceManagerPostgreSqlContext.Default);
            return await Task.FromResult(new PostgreSqlFirewallRuleResource(_client, data)).ConfigureAwait(false);
        }
    }
}
