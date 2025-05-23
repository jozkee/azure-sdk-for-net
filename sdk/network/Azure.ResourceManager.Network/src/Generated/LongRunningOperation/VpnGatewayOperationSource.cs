// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System.ClientModel.Primitives;
using System.Threading;
using System.Threading.Tasks;
using Azure.Core;

namespace Azure.ResourceManager.Network
{
    internal class VpnGatewayOperationSource : IOperationSource<VpnGatewayResource>
    {
        private readonly ArmClient _client;

        internal VpnGatewayOperationSource(ArmClient client)
        {
            _client = client;
        }

        VpnGatewayResource IOperationSource<VpnGatewayResource>.CreateResult(Response response, CancellationToken cancellationToken)
        {
            var data = ModelReaderWriter.Read<VpnGatewayData>(response.Content, ModelReaderWriterOptions.Json, AzureResourceManagerNetworkContext.Default);
            return new VpnGatewayResource(_client, data);
        }

        async ValueTask<VpnGatewayResource> IOperationSource<VpnGatewayResource>.CreateResultAsync(Response response, CancellationToken cancellationToken)
        {
            var data = ModelReaderWriter.Read<VpnGatewayData>(response.Content, ModelReaderWriterOptions.Json, AzureResourceManagerNetworkContext.Default);
            return await Task.FromResult(new VpnGatewayResource(_client, data)).ConfigureAwait(false);
        }
    }
}
