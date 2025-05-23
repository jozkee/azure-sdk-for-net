// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.Collections.Generic;
using Azure.Core;

namespace Azure.ResourceManager.DependencyMap.Models
{
    /// <summary>
    /// The properties of Discovery Source resource
    /// Please note <see cref="DependencyMapDiscoverySourceProperties"/> is the base class. According to the scenario, a derived class of the base class might need to be assigned here, or this property needs to be casted to one of the possible derived classes.
    /// The available derived classes include <see cref="OffAzureDiscoverySourceProperties"/>.
    /// </summary>
    public abstract partial class DependencyMapDiscoverySourceProperties
    {
        /// <summary>
        /// Keeps track of any properties unknown to the library.
        /// <para>
        /// To assign an object to the value of this property use <see cref="BinaryData.FromObjectAsJson{T}(T, System.Text.Json.JsonSerializerOptions?)"/>.
        /// </para>
        /// <para>
        /// To assign an already formatted json string to this property use <see cref="BinaryData.FromString(string)"/>.
        /// </para>
        /// <para>
        /// Examples:
        /// <list type="bullet">
        /// <item>
        /// <term>BinaryData.FromObjectAsJson("foo")</term>
        /// <description>Creates a payload of "foo".</description>
        /// </item>
        /// <item>
        /// <term>BinaryData.FromString("\"foo\"")</term>
        /// <description>Creates a payload of "foo".</description>
        /// </item>
        /// <item>
        /// <term>BinaryData.FromObjectAsJson(new { key = "value" })</term>
        /// <description>Creates a payload of { "key": "value" }.</description>
        /// </item>
        /// <item>
        /// <term>BinaryData.FromString("{\"key\": \"value\"}")</term>
        /// <description>Creates a payload of { "key": "value" }.</description>
        /// </item>
        /// </list>
        /// </para>
        /// </summary>
        private protected IDictionary<string, BinaryData> _serializedAdditionalRawData;

        /// <summary> Initializes a new instance of <see cref="DependencyMapDiscoverySourceProperties"/>. </summary>
        /// <param name="sourceId"> Source ArmId of Discovery Source resource. </param>
        /// <exception cref="ArgumentNullException"> <paramref name="sourceId"/> is null. </exception>
        protected DependencyMapDiscoverySourceProperties(ResourceIdentifier sourceId)
        {
            Argument.AssertNotNull(sourceId, nameof(sourceId));

            SourceId = sourceId;
        }

        /// <summary> Initializes a new instance of <see cref="DependencyMapDiscoverySourceProperties"/>. </summary>
        /// <param name="provisioningState"> Provisioning state of Discovery Source resource. </param>
        /// <param name="sourceType"> Source type of Discovery Source resource. </param>
        /// <param name="sourceId"> Source ArmId of Discovery Source resource. </param>
        /// <param name="serializedAdditionalRawData"> Keeps track of any properties unknown to the library. </param>
        internal DependencyMapDiscoverySourceProperties(DependencyMapProvisioningState? provisioningState, SourceType sourceType, ResourceIdentifier sourceId, IDictionary<string, BinaryData> serializedAdditionalRawData)
        {
            ProvisioningState = provisioningState;
            SourceType = sourceType;
            SourceId = sourceId;
            _serializedAdditionalRawData = serializedAdditionalRawData;
        }

        /// <summary> Initializes a new instance of <see cref="DependencyMapDiscoverySourceProperties"/> for deserialization. </summary>
        internal DependencyMapDiscoverySourceProperties()
        {
        }

        /// <summary> Provisioning state of Discovery Source resource. </summary>
        public DependencyMapProvisioningState? ProvisioningState { get; }
        /// <summary> Source type of Discovery Source resource. </summary>
        internal SourceType SourceType { get; set; }
        /// <summary> Source ArmId of Discovery Source resource. </summary>
        public ResourceIdentifier SourceId { get; set; }
    }
}
