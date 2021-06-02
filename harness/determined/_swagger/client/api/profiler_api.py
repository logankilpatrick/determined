# coding: utf-8

"""
    Determined API (Beta)

    Determined helps deep learning teams train models more quickly, easily share GPU resources, and effectively collaborate. Determined allows deep learning engineers to focus on building and training models at scale, without needing to worry about DevOps or writing custom code for common tasks like fault tolerance or experiment tracking.  You can think of Determined as a platform that bridges the gap between tools like TensorFlow and PyTorch --- which work great for a single researcher with a single GPU --- to the challenges that arise when doing deep learning at scale, as teams, clusters, and data sets all increase in size.  # noqa: E501

    OpenAPI spec version: 0.1
    Contact: community@determined.ai
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


from __future__ import absolute_import

import re  # noqa: F401

# python 2 and python 3 compatibility library
import six

from determined._swagger.client.api_client import ApiClient


class ProfilerApi(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    Ref: https://github.com/swagger-api/swagger-codegen
    """

    def __init__(self, api_client=None):
        if api_client is None:
            api_client = ApiClient()
        self.api_client = api_client

    def determined_get_trial_profiler_available_series(self, trial_id, **kwargs):  # noqa: E501
        """Stream the available series in a trial's profiler metrics.  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.determined_get_trial_profiler_available_series(trial_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int trial_id: The requested trial's id. (required)
        :return: StreamResultOfV1GetTrialProfilerAvailableSeriesResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.determined_get_trial_profiler_available_series_with_http_info(trial_id, **kwargs)  # noqa: E501
        else:
            (data) = self.determined_get_trial_profiler_available_series_with_http_info(trial_id, **kwargs)  # noqa: E501
            return data

    def determined_get_trial_profiler_available_series_with_http_info(self, trial_id, **kwargs):  # noqa: E501
        """Stream the available series in a trial's profiler metrics.  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.determined_get_trial_profiler_available_series_with_http_info(trial_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int trial_id: The requested trial's id. (required)
        :return: StreamResultOfV1GetTrialProfilerAvailableSeriesResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['trial_id']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method determined_get_trial_profiler_available_series" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'trial_id' is set
        if ('trial_id' not in params or
                params['trial_id'] is None):
            raise ValueError("Missing the required parameter `trial_id` when calling `determined_get_trial_profiler_available_series`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'trial_id' in params:
            path_params['trialId'] = params['trial_id']  # noqa: E501

        query_params = []

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['BearerToken']  # noqa: E501

        return self.api_client.call_api(
            '/api/v1/trials/{trialId}/profiler/available_series', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='StreamResultOfV1GetTrialProfilerAvailableSeriesResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)

    def determined_get_trial_profiler_metrics(self, labels_trial_id, **kwargs):  # noqa: E501
        """Stream trial profiler metrics.  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.determined_get_trial_profiler_metrics(labels_trial_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int labels_trial_id: The ID of the trial. (required)
        :param str labels_name: The name of the metric.
        :param str labels_agent_id: The agent ID associated with the metric.
        :param str labels_gpu_uuid: The GPU UUID associated with the metric.
        :param str labels_metric_type: The type of the metric.   - PROFILER_METRIC_TYPE_UNSPECIFIED: Zero-value (not allowed).  - PROFILER_METRIC_TYPE_SYSTEM: For systems metrics, like GPU utilization or memory.  - PROFILER_METRIC_TYPE_TIMING: For timing metrics, like how long a backwards pass or getting a batch from the dataloader took.
        :return: StreamResultOfV1GetTrialProfilerMetricsResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('async_req'):
            return self.determined_get_trial_profiler_metrics_with_http_info(labels_trial_id, **kwargs)  # noqa: E501
        else:
            (data) = self.determined_get_trial_profiler_metrics_with_http_info(labels_trial_id, **kwargs)  # noqa: E501
            return data

    def determined_get_trial_profiler_metrics_with_http_info(self, labels_trial_id, **kwargs):  # noqa: E501
        """Stream trial profiler metrics.  # noqa: E501

        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.determined_get_trial_profiler_metrics_with_http_info(labels_trial_id, async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param int labels_trial_id: The ID of the trial. (required)
        :param str labels_name: The name of the metric.
        :param str labels_agent_id: The agent ID associated with the metric.
        :param str labels_gpu_uuid: The GPU UUID associated with the metric.
        :param str labels_metric_type: The type of the metric.   - PROFILER_METRIC_TYPE_UNSPECIFIED: Zero-value (not allowed).  - PROFILER_METRIC_TYPE_SYSTEM: For systems metrics, like GPU utilization or memory.  - PROFILER_METRIC_TYPE_TIMING: For timing metrics, like how long a backwards pass or getting a batch from the dataloader took.
        :return: StreamResultOfV1GetTrialProfilerMetricsResponse
                 If the method is called asynchronously,
                 returns the request thread.
        """

        all_params = ['labels_trial_id', 'labels_name', 'labels_agent_id', 'labels_gpu_uuid', 'labels_metric_type']  # noqa: E501
        all_params.append('async_req')
        all_params.append('_return_http_data_only')
        all_params.append('_preload_content')
        all_params.append('_request_timeout')

        params = locals()
        for key, val in six.iteritems(params['kwargs']):
            if key not in all_params:
                raise TypeError(
                    "Got an unexpected keyword argument '%s'"
                    " to method determined_get_trial_profiler_metrics" % key
                )
            params[key] = val
        del params['kwargs']
        # verify the required parameter 'labels_trial_id' is set
        if ('labels_trial_id' not in params or
                params['labels_trial_id'] is None):
            raise ValueError("Missing the required parameter `labels_trial_id` when calling `determined_get_trial_profiler_metrics`")  # noqa: E501

        collection_formats = {}

        path_params = {}
        if 'labels_trial_id' in params:
            path_params['labels.trialId'] = params['labels_trial_id']  # noqa: E501

        query_params = []
        if 'labels_name' in params:
            query_params.append(('labels.name', params['labels_name']))  # noqa: E501
        if 'labels_agent_id' in params:
            query_params.append(('labels.agentId', params['labels_agent_id']))  # noqa: E501
        if 'labels_gpu_uuid' in params:
            query_params.append(('labels.gpuUuid', params['labels_gpu_uuid']))  # noqa: E501
        if 'labels_metric_type' in params:
            query_params.append(('labels.metricType', params['labels_metric_type']))  # noqa: E501

        header_params = {}

        form_params = []
        local_var_files = {}

        body_params = None
        # HTTP header `Accept`
        header_params['Accept'] = self.api_client.select_header_accept(
            ['application/json'])  # noqa: E501

        # HTTP header `Content-Type`
        header_params['Content-Type'] = self.api_client.select_header_content_type(  # noqa: E501
            ['application/json'])  # noqa: E501

        # Authentication setting
        auth_settings = ['BearerToken']  # noqa: E501

        return self.api_client.call_api(
            '/api/v1/trials/{labels.trialId}/profiler/metrics', 'GET',
            path_params,
            query_params,
            header_params,
            body=body_params,
            post_params=form_params,
            files=local_var_files,
            response_type='StreamResultOfV1GetTrialProfilerMetricsResponse',  # noqa: E501
            auth_settings=auth_settings,
            async_req=params.get('async_req'),
            _return_http_data_only=params.get('_return_http_data_only'),
            _preload_content=params.get('_preload_content', True),
            _request_timeout=params.get('_request_timeout'),
            collection_formats=collection_formats)