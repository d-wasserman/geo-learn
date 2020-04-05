# --------------------------------
# Name: gelearnlib.py
# Purpose: This script is a function library for geolearn script examples.
# Current Owner: David Wasserman
# Last Modified: 4/5/2020
# Copyright:   (c) David Wasserman
# ArcGIS Version:   ArcGIS Pro
# Python Version:   3.6
# --------------------------------
# Copyright 2016 David J. Wasserman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------
# Import Modules
import arcpy
import numpy as np
import os
from scipy import stats

try:
    import pandas as pd
except:
    arcpy.AddError("This library requires Pandas installed in the ArcGIS Python Install."
                   " Might require installing pre-requisite libraries and software.")


# Function Definitions
######################
def generate_percentile_metric(dataframe, fields_to_score, method="max", na_fill=.5, invert=False, pct=True):
    """When passed a dataframe and fields to score, this function will return a percentile score (pct rank) based on the
    settings passed to the function including how to fill in na values or whether to invert the metric.
    :param dataframe: dataframe that will be returned with new scored fields
    :param fields_to_score: list of columns to score
    :param method: {‘average’, ‘min’, ‘max’, ‘first’, ‘dense’}
        average: average rank of group
        min: lowest rank in group
        max: highest rank in group
        first: ranks assigned in order they appear in the array
        dense: like ‘min’, but rank always increases by 1 between groups
    :na_fill: float
        Will fill kept null values with the chosen value. Defaults to .5
    :invert : boolean
        Will make lower values be scored
    pct:  boolean, default True
        Computes percentage rank of data"""
    for field in fields_to_score:
        try:
            new_score = "{0}_Score".format(field)
            if not invert:
                dataframe[new_score] = dataframe[field].rank(method=method, pct=pct).fillna(value=na_fill)
            else:
                dataframe[new_score] = dataframe[field].rank(method=method, pct=pct, ascending=False).fillna(
                    value=na_fill)
        except:
            print("WARNING:Could not score column {0}. Check input dataframe.".format(field))
    return dataframe


# Functions pulled from ShareArcLib - ArcNumerical Github: d-wasserman
def func_report(function=None, reportBool=False):
    """This decorator function is designed to be used as a wrapper with other functions to enable basic try and except
     reporting (if function fails it will report the name of the function that failed and its arguments. If a report
      boolean is true the function will report inputs and outputs of a function.-David Wasserman"""

    def func_report_decorator(function):
        def func_wrapper(*args, **kwargs):
            try:
                func_result = function(*args, **kwargs)
                if reportBool:
                    print("Function:{0}".format(str(function.__name__)))
                    print("     Input(s):{0}".format(str(args)))
                    print("     Ouput(s):{0}".format(str(func_result)))
                return func_result
            except Exception as e:
                print(
                    "{0} - function failed -|- Function arguments were:{1}.".format(str(function.__name__), str(args)))
                print(e.args[0])

        return func_wrapper

    if not function:  # User passed in a bool argument
        def waiting_for_function(function):
            return func_report_decorator(function)

        return waiting_for_function
    else:
        return func_report_decorator(function)


def arc_tool_report(function=None, arcToolMessageBool=False, arcProgressorBool=False):
    """This decorator function is designed to be used as a wrapper with other GIS functions to enable basic try and except
     reporting (if function fails it will report the name of the function that failed and its arguments. If a report
      boolean is true the function will report inputs and outputs of a function.-David Wasserman"""

    def arc_tool_report_decorator(function):
        def func_wrapper(*args, **kwargs):
            try:
                func_result = function(*args, **kwargs)
                if arcToolMessageBool:
                    arcpy.AddMessage("Function:{0}".format(str(function.__name__)))
                    arcpy.AddMessage("     Input(s):{0}".format(str(args)))
                    arcpy.AddMessage("     Ouput(s):{0}".format(str(func_result)))
                if arcProgressorBool:
                    arcpy.SetProgressorLabel("Function:{0}".format(str(function.__name__)))
                    arcpy.SetProgressorLabel("     Input(s):{0}".format(str(args)))
                    arcpy.SetProgressorLabel("     Ouput(s):{0}".format(str(func_result)))
                return func_result
            except Exception as e:
                arcpy.AddMessage(
                    "{0} - function failed -|- Function arguments were:{1}.".format(str(function.__name__),
                                                                                    str(args)))
                print(
                    "{0} - function failed -|- Function arguments were:{1}.".format(str(function.__name__), str(args)))
                print(e.args[0])

        return func_wrapper

    if not function:  # User passed in a bool argument
        def waiting_for_function(function):
            return arc_tool_report_decorator(function)

        return waiting_for_function
    else:
        return arc_tool_report_decorator(function)


@arc_tool_report
def arc_print(string, progressor_Bool=False):
    """ This function is used to simplify using arcpy reporting for tool creation,if progressor bool is true it will
    create a tool label."""
    casted_string = str(string)
    if progressor_Bool:
        arcpy.SetProgressorLabel(casted_string)
        arcpy.AddMessage(casted_string)
        print(casted_string)
    else:
        arcpy.AddMessage(casted_string)
        print(casted_string)


@arc_tool_report
def field_exist(featureclass, fieldname):
    """ArcFunction
     Check if a field in a feature class field exists and return true it does, false if not.- David Wasserman"""
    fieldList = arcpy.ListFields(featureclass, fieldname)
    fieldCount = len(fieldList)
    if (fieldCount >= 1) and fieldname.strip():  # If there is one or more of this field return true
        return True
    else:
        return False


@arc_tool_report
def add_new_field(in_table, field_name, field_type, field_precision="#", field_scale="#", field_length="#",
                  field_alias="#", field_is_nullable="#", field_is_required="#", field_domain="#"):
    """ArcFunction
    Add a new field if it currently does not exist. Add field alone is slower than checking first.- David Wasserman"""
    if field_exist(in_table, field_name):
        print(field_name + " Exists")
        arcpy.AddMessage(field_name + " Exists")
    else:
        print("Adding " + field_name)
        arcpy.AddMessage("Adding " + field_name)
        arcpy.AddField_management(in_table, field_name, field_type, field_precision, field_scale,
                                  field_length,
                                  field_alias,
                                  field_is_nullable, field_is_required, field_domain)


@arc_tool_report
def validate_df_names(dataframe, output_feature_class_workspace):
    """Returns pandas dataframe with all col names renamed to be valid arcgis table names."""
    new_name_list = []
    old_names = dataframe.columns.names
    for name in old_names:
        new_name = arcpy.ValidateFieldName(name, output_feature_class_workspace)
        new_name_list.append(new_name)
    rename_dict = {i: j for i, j in zip(old_names, new_name_list)}
    dataframe.rename(index=str, columns=rename_dict)
    return dataframe


@arc_tool_report
def arcgis_table_to_df(in_fc, input_fields=None, query=""):
    """Function will convert an arcgis table into a pandas dataframe with an object ID index, and the selected
    input fields using an arcpy.da.SearchCursor.
    :param - in_fc - input feature class or table to convert
    :param - input_fields - fields to input to a da search cursor for retrieval
    :param - query - sql query to grab appropriate values
    :returns - pandas.DataFrame"""
    OIDFieldName = arcpy.Describe(in_fc).OIDFieldName
    if input_fields:
        final_fields = [OIDFieldName] + input_fields
    else:
        final_fields = [field.name for field in arcpy.ListFields(in_fc)]
    data = [row for row in arcpy.da.SearchCursor(in_fc, final_fields, where_clause=query)]
    fc_dataframe = pd.DataFrame(data, columns=final_fields)
    fc_dataframe = fc_dataframe.set_index(OIDFieldName, drop=True)
    return fc_dataframe


@arc_tool_report
def arcgis_table_to_dataframe(in_fc, input_fields, query="", skip_nulls=False, null_values=None):
    """Function will convert an arcgis table into a pandas dataframe with an object ID index, and the selected
    input fields. Uses TableToNumPyArray to get initial data.
    :param - in_fc - input feature class or table to convert
    :param - input_fields - fields to input into a da numpy converter function
    :param - query - sql like query to filter out records returned
    :param - skip_nulls - skip rows with null values
    :param - null_values - values to replace null values with.
    :returns - pandas dataframe"""
    OIDFieldName = arcpy.Describe(in_fc).OIDFieldName
    if input_fields:
        final_fields = [OIDFieldName] + input_fields
    else:
        final_fields = [field.name for field in arcpy.ListFields(in_fc)]
    np_array = arcpy.da.TableToNumPyArray(in_fc, final_fields, query, skip_nulls, null_values)
    object_id_index = np_array[OIDFieldName]
    fc_dataframe = pd.DataFrame(np_array, index=object_id_index, columns=input_fields)
    return fc_dataframe


@arc_tool_report
def arc_unique_values(table, field, filter_falsy=False):
    """This function will return a list of unique values from a passed field. If the optional bool is true,
    this function will scrub out null/falsy values. """
    with arcpy.da.SearchCursor(table, [field]) as cursor:
        if filter_falsy:
            return sorted({row[0] for row in cursor if row[0]})
        else:
            return sorted({row[0] for row in cursor})


def generate_statistical_fieldmap(target_features, join_features, prepended_name="", merge_rule_dict={}):
    """Generates field map object based on passed field objects based on passed tables (list),
    input_field_objects (list), and passed statistics fields to choose for numeric and categorical variables. Output
    fields take the form of *merge rule*+*prepended_name*+*fieldname*.
    :param target_features(str)- target feature class that will maintain its field attributes
    :param join_features(str)- join feature class whose numeric fields will be joined based on the merge rule dictionary
    :param prepended_name(str)- modifies output join fields with param text between the statistics and the original field name
    :param (dict)- a  dictionary of the form {statistic_type:[Fields,To,Summarize]}
    :returns arcpy field mapping object"""
    field_mappings = arcpy.FieldMappings()
    # We want every field in 'target_features' and all fields in join_features that are present
    # in the field statistics mappping.
    field_mappings.addTable(target_features)
    for merge_rule in merge_rule_dict:
        for field in merge_rule_dict[merge_rule]:
            new_field_map = arcpy.FieldMap()
            new_field_map.addInputField(join_features, field)
            new_field_map.mergeRule = merge_rule
            out_field = new_field_map.outputField
            out_field.name = str(merge_rule) + str(prepended_name) + str(field)
            out_field.aliasName = str(merge_rule) + str(prepended_name) + str(field)
            new_field_map.outputField = out_field
            field_mappings.addFieldMap(new_field_map)
    return field_mappings


@arc_tool_report
def determine_extract_and_subset_fields(input_feature_class, all_fields, exception_fields=[], additional_fields=[],
                                        subset_removal_fields=[]):
    """This worker funciton will create two sets of fields, one set to be passed on to fit/processing methods if they
    exist in a feature class or if they are designated to be removed. The goal of this function is to get one list to
    build the main numpy array from the feature class, and another set to be passed to the fit/processing methods.
    The subset removal field helps filter out weight fields that need to be in the extract fields"""
    extract_fields = []
    subset_fields = []
    ignore_removal = True
    valid_fields = [str(i) for i in all_fields if field_exist(input_feature_class, str(i))]
    valid_additional_fields = [str(i) for i in additional_fields if field_exist(input_feature_class, str(i))]
    if len(valid_fields) >= 1:
        extract_fields = valid_additional_fields + valid_fields
        subset_fields = valid_fields
    else:
        extract_fields = valid_additional_fields + exception_fields
        subset_fields = exception_fields
        ignore_removal = False
    if ignore_removal:
        for subset_removal in subset_removal_fields:
            if subset_removal in subset_fields:
                subset_fields.remove(subset_removal)
    return extract_fields, subset_fields


@arc_tool_report
def validate_weight_list(sample_weight, n_samples):
    """This will return a valid weight array based on a passed sample weight array and the length/shape of the sample
    features or it will return None and return a flag boolean to indicate not to weight the sample."""
    use_weighted = True
    if sample_weight is None:
        # uniform sample weights-no change to input data.
        sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        use_weighted = False
    else:
        # user-provided array
        sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                   order="C")
    if sample_weight.shape[0] != n_samples:
        raise ValueError("Shape of features and sample_weight do not match.")
    return sample_weight, use_weighted


@arc_tool_report
def return_weighted_array(dataset, weightlist):
    """This function will take a dataset iterable and weight array and create a new numpy array
     with the components repeated based on the corresponding weight field. The weight field list will be validated. """
    validated_weights, use_weightlist = validate_weight_list(weightlist, int(len(dataset)))
    if use_weightlist:
        weighted_array = np.repeat(dataset.values, weightlist, axis=0)
    else:
        weighted_array = dataset
    return weighted_array


@arc_tool_report
def reduce_weighted_array(weighted_array, weight_list):
    """Reduce weighted repeated array to original feature size based on the pass weighted array and the last weighted
    record kept in the created weighted array. Cumulative sum is used to derive index locations of last elements of a
     weight list assume features at same record share an index."""
    indices = np.cumsum(weight_list) - 1
    reduced_array = np.take(weighted_array, indices=indices)  # original locations
    return reduced_array


# End do_analysis function

# This test allows the script to be used from the operating
# system command prompt (stand-alone), in a Python IDE,
# as a geoprocessing script tool, or as a module imported in
# another script
if __name__ == '__main__':
    print("Geo-Learn Python Lib Script. No additional functions executed.")
