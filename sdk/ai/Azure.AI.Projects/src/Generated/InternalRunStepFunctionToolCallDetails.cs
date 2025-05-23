// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// <auto-generated/>

#nullable disable

using System;
using System.Collections.Generic;

namespace Azure.AI.Projects
{
    /// <summary> The detailed information about the function called by the model. </summary>
    internal partial class InternalRunStepFunctionToolCallDetails
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
        private IDictionary<string, BinaryData> _serializedAdditionalRawData;

        /// <summary> Initializes a new instance of <see cref="InternalRunStepFunctionToolCallDetails"/>. </summary>
        /// <param name="name"> The name of the function. </param>
        /// <param name="arguments"> The arguments that the model requires are provided to the named function. </param>
        /// <param name="output"> The output of the function, only populated for function calls that have already have had their outputs submitted. </param>
        /// <exception cref="ArgumentNullException"> <paramref name="name"/> or <paramref name="arguments"/> is null. </exception>
        internal InternalRunStepFunctionToolCallDetails(string name, string arguments, string output)
        {
            Argument.AssertNotNull(name, nameof(name));
            Argument.AssertNotNull(arguments, nameof(arguments));

            Name = name;
            Arguments = arguments;
            Output = output;
        }

        /// <summary> Initializes a new instance of <see cref="InternalRunStepFunctionToolCallDetails"/>. </summary>
        /// <param name="name"> The name of the function. </param>
        /// <param name="arguments"> The arguments that the model requires are provided to the named function. </param>
        /// <param name="output"> The output of the function, only populated for function calls that have already have had their outputs submitted. </param>
        /// <param name="serializedAdditionalRawData"> Keeps track of any properties unknown to the library. </param>
        internal InternalRunStepFunctionToolCallDetails(string name, string arguments, string output, IDictionary<string, BinaryData> serializedAdditionalRawData)
        {
            Name = name;
            Arguments = arguments;
            Output = output;
            _serializedAdditionalRawData = serializedAdditionalRawData;
        }

        /// <summary> Initializes a new instance of <see cref="InternalRunStepFunctionToolCallDetails"/> for deserialization. </summary>
        internal InternalRunStepFunctionToolCallDetails()
        {
        }

        /// <summary> The name of the function. </summary>
        public string Name { get; }
        /// <summary> The arguments that the model requires are provided to the named function. </summary>
        public string Arguments { get; }
        /// <summary> The output of the function, only populated for function calls that have already have had their outputs submitted. </summary>
        public string Output { get; }
    }
}
