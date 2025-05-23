// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System.ClientModel.Primitives;
using System.Threading;
using System.Threading.Tasks;
using Azure.Core;

namespace Azure.ResourceManager.Redis
{
    internal class RedisCacheAccessPolicyOperationSource : IOperationSource<RedisCacheAccessPolicyResource>
    {
        private readonly ArmClient _client;

        internal RedisCacheAccessPolicyOperationSource(ArmClient client)
        {
            _client = client;
        }

        RedisCacheAccessPolicyResource IOperationSource<RedisCacheAccessPolicyResource>.CreateResult(Response response, CancellationToken cancellationToken)
        {
            var data = ModelReaderWriter.Read<RedisCacheAccessPolicyData>(response.Content, ModelReaderWriterOptions.Json, AzureResourceManagerRedisContext.Default);
            return new RedisCacheAccessPolicyResource(_client, data);
        }

        async ValueTask<RedisCacheAccessPolicyResource> IOperationSource<RedisCacheAccessPolicyResource>.CreateResultAsync(Response response, CancellationToken cancellationToken)
        {
            var data = ModelReaderWriter.Read<RedisCacheAccessPolicyData>(response.Content, ModelReaderWriterOptions.Json, AzureResourceManagerRedisContext.Default);
            return await Task.FromResult(new RedisCacheAccessPolicyResource(_client, data)).ConfigureAwait(false);
        }
    }
}
