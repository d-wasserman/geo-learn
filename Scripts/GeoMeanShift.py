# --------------------------------
# Name: GeoMeanShift.py
# Purpose: This script is intended to allow ArcGIS users that have Scikit Learn installed in their python installation
# utilize Mean Shift to create clusters of geographic features based on their centroids.
# Current Owner: David Wasserman
# Last Modified: 11/01/2016
# Copyright:   (c) CoAdapt
# ArcGIS Version:   ArcGIS Pro/10.4
# Python Version:   3.5/2.7
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


import os, arcpy, itertools
import numpy as np

try:
    from sklearn import cluster
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
except:
    arcpy.AddError("This library requires Sci-kit Learn installed in the ArcGIS Python Install."
                   " Might require installing pre-requisite libraries and software.")

# Define input parameters

input_feature_class = arcpy.GetParameterAsText(0)
bandwidth = arcpy.GetParameter(1)
use_estimated_bandwidth = arcpy.GetParameter(2)
cluster_all_points = arcpy.GetParameter(3)
output_feature_class_centroids = arcpy.GetParameterAsText(4)
weight_field  = arcpy.GetParameterAsText(5)
fields_to_cluster = str(arcpy.GetParameterAsText(6)).split(";")



# Function Definitions
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
    if (fieldCount >= 1)and fieldname.strip():  # If there is one or more of this field return true
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
def get_fields(featureClass, excludedTolkens=["OID", "Geometry"],
               excludedFields=["shape_area", "shape_length"]):
    """Get all field names from an incoming feature class defaulting to excluding tolkens and shape area & length.
    Inputs: Feature class, excluding tokens list, excluded fields list.
    Outputs: List of field names from input feature class. """
    try:
        try:  # If  A feature Class split to game name
            fcName = os.path.split(featureClass)[1]
        except:  # If a Feature Layer, just print the Layer Name
            fcName = featureClass
        field_list = [f.name for f in arcpy.ListFields(featureClass) if f.type not in excludedTolkens
                      and f.name.lower() not in excludedFields]
        arc_print("The field list for {0} is:{1}".format(str(fcName), str(field_list)), True)
        return field_list
    except:
        arc_print(
            "Could not get fields for the following input {0}, returned an empty list.".format(
                str(featureClass)),
            True)
        arcpy.AddWarning(
            "Could not get fields for the following input {0}, returned an empty list.".format(
                str(featureClass)))
        field_list = []
        return field_list

@arc_tool_report
def determine_extract_and_subset_fields(input_feature_class,all_fields,exception_fields=[],additional_fields=[],
                                        subset_removal_fields=[]):
    """This worker funciton will create two sets of fields, one set to be passed on to fit/processing methods if they
    exist in a feature class or if they are designated to be removed. The goal of this function is to get one list to
    build the main numpy array from the feature class, and another set to be passed to the fit/processing methods.
    The subset removal field helps filter out weight fields that need to be in the extract fields"""
    extract_fields=[]
    subset_fields=[]
    ignore_removal=True
    valid_fields=[str(i) for i in all_fields if field_exist(input_feature_class,str(i))]
    valid_additional_fields=[str(i) for i in additional_fields if field_exist(input_feature_class,str(i))]
    if len(valid_fields)>=1:
        extract_fields=valid_additional_fields+valid_fields
        subset_fields= valid_fields
    else:
        extract_fields = valid_additional_fields+exception_fields
        subset_fields=exception_fields
        ignore_removal=False
    if ignore_removal:
        for subset_removal in subset_removal_fields:
            if subset_removal in subset_fields:
                subset_fields.remove(subset_removal)
    return extract_fields,subset_fields

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
        weighted_array = np.repeat(dataset, weightlist, axis=0)
    else:
        weighted_array = dataset
    return weighted_array

@arc_tool_report
def reduce_weighted_array(weighted_array,weight_list):
    """Reduce weighted repeated array to original feature size based on the pass weighted array and the last weighted
    record kept in the created weighted array. Cumulative sum is used to derive index locations of last elements of a
     weight list assume features at same record share an index."""
    indices = np.cumsum(weight_list) - 1
    reduced_array = np.take(weighted_array, indices=indices)  # original locations
    return reduced_array

# Function Definitions
def classify_features_meanshift(in_fc, search_radius, output_fc,weight_field=None,alternative_fields=[],
                                bin_seeding=False, min_bin_freq=1,cluster_all_pts=True,estimate_bandwidth=False):
    """Take in a feature class of points and classify them into clusters using Mean Shift clustering from Scikit learn.
     Append field labels to the input feature class using Extend Numpy Array function."""
    try:
        # Declare Starting Variables
        desc = arcpy.Describe(in_fc)
        SpatialReference = desc.spatialReference
        workspace = os.path.dirname(desc.catalogPath)
        arc_print("Converting '{0}' feature class to numpy array based on inputs.".format(str(desc.name)))
        centroid = 'SHAPE@XY'
        OIDFieldName = desc.OIDFieldName
        feature_class_fields,cluster_fields=determine_extract_and_subset_fields(in_fc,
                            alternative_fields,[centroid],[OIDFieldName,weight_field],[weight_field])
        arc_print("Feature class clustering will be conducted on the following fields: {0}".format(cluster_fields))
        # Convert Feature Class to NP array
        geoarray = arcpy.da.FeatureClassToNumPyArray(in_fc, feature_class_fields,
                                                     null_value=1)  # Null Values of treated as one feature -weight
        data= geoarray[cluster_fields[0]]
        #Create Weighted arrays if weight field is present.
        using_cluster_weight= True if weight_field in feature_class_fields else False
        if using_cluster_weight:
            arc_print("Preparing weighted Data for clustering.")
            data= return_weighted_array(data,geoarray[weight_field])
        # Standardize Data if using Fields.
        clustering_on_geometry= True if centroid in cluster_fields else False
        if not clustering_on_geometry: # If Clustering on arbitrary fields, standardize data.
            arc_print("Processing arbitrary fields rather than feature coordinates. Standardizing data with Sklearn's "
                     "StandardScaler(). Bandwidth should be in standardized units or using the estimated bandwidth.")
            scaler=StandardScaler().fit(data)
            data_to_cluster = scaler.transform(data)
        else:
            data_to_cluster = data
        # Estimate Bandwidth if chosen.
        if estimate_bandwidth or search_radius<=0.0:
            search_radius = cluster.estimate_bandwidth(data_to_cluster)
            arc_print("Using estimated bandwidth of {0} based on estimation function.".format(search_radius), True)
        arc_print("Using geographic coordinates to classify with Mean_Shift.", True)
        meanshift_classification = cluster.MeanShift(bandwidth=search_radius, bin_seeding=bin_seeding,
                                                     min_bin_freq=min_bin_freq, cluster_all=cluster_all_pts).fit(
                data_to_cluster)
        cluster_centroids = meanshift_classification.cluster_centers_
        labels = meanshift_classification.labels_
        # Number of clusters in labels, ignoring noise if present.
        unique_clusters = set([i for i in labels if i != -1])
        cluster_count = len(unique_clusters)
        arc_print('Estimated number of clusters: {0}'.format(cluster_count), True)
        try:
            arc_print("Silhouette Coefficient: {0}.".format(metrics.silhouette_score(data_to_cluster, labels)), True)
            arc_print(
                    """Wikipedia: The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.""")
        except Exception as e:
            arc_print("Could not compute Silhouette Coefficient. Error: {0}".format(str(e.args[0])), True)
        #After Clustering and Metric gathering extend feature class and export.
        arc_print("Appending Labels from Mean Shift to new numpy array.", True)
        JoinField = str(arcpy.ValidateFieldName("NPIndexJoin", workspace))
        LabelField = str(arcpy.ValidateFieldName("MeanShiftLabel", workspace))
        LabelCount = str(arcpy.ValidateFieldName("LabelCount", workspace))
        ShapeXField = str(arcpy.ValidateFieldName("ShapeX", workspace))
        ShapeYField = str(arcpy.ValidateFieldName("ShapeY", workspace))
        finalMean_ShiftArray = np.array(list(zip(geoarray[OIDFieldName], labels)),
                                        dtype=[(JoinField, np.int32), (LabelField, np.int32)])
        arc_print("Extending Label Fields to Output Feature Class. Clusters labels start at 0, noise is labeled -1.",
                 True)
        if using_cluster_weight:
            labels=reduce_weighted_array(labels,geoarray[weight_field])
        arcpy.da.ExtendTable(in_fc, OIDFieldName, finalMean_ShiftArray, JoinField, append_only=False)
        #Export feature class centroids
        directory_name = os.path.split(output_fc)[0]
        file_name = os.path.split(output_fc)[1]
        if arcpy.Exists(directory_name) and clustering_on_geometry:
            #Only create new feature class it output locations exists and if there clustering is on geometry.
            arc_print("Creating Centroid Feature Class of clusters {0}.".format(str(file_name)), True)
            ShapeX, ShapeY = zip(*cluster_centroids)
            count_of_items_per_label = [int(labels.tolist().count(unique_value)) for unique_value in unique_clusters]
            final_centroid_array = np.asarray(list(zip(ShapeX, ShapeY, unique_clusters, count_of_items_per_label)),
                                              dtype=[(ShapeXField, np.float64), (ShapeYField, np.float64),
                                                     (LabelField, np.int32), (LabelCount, np.int32)])
            arcpy.da.NumPyArrayToFeatureClass(final_centroid_array, output_fc, (ShapeXField, ShapeYField),
                                              SpatialReference)
        del geoarray, finalMean_ShiftArray, labels, meanshift_classification
        arc_print("Script Completed Successfully.", True)
    except arcpy.ExecuteError:
        arc_print(arcpy.GetMessages(2))
    except Exception as e:
        arc_print(e.args[0])


# End do_analysis function

# This test allows the script to be used from the operating
# system command prompt (stand-alone), in a Python IDE,
# as a geoprocessing script tool, or as a module imported in
# another script
if __name__ == '__main__':
    classify_features_meanshift(input_feature_class, weight_field=weight_field,
                                search_radius=bandwidth, output_fc=output_feature_class_centroids,
                                estimate_bandwidth=use_estimated_bandwidth, cluster_all_pts=cluster_all_points)
