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
import os, arcpy
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


# Todo Add output feature class for centroids. #output_feature_class_centroids=arcpy.GetParameterAsText(4)

# Function Definitions
def funcReport(function=None, reportBool=False):
    """This decorator function is designed to be used as a wrapper with other functions to enable basic try and except
     reporting (if function fails it will report the name of the function that failed and its arguments. If a report
      boolean is true the function will report inputs and outputs of a function.-David Wasserman"""

    def funcReport_Decorator(function):
        def funcWrapper(*args, **kwargs):
            try:
                funcResult = function(*args, **kwargs)
                if reportBool:
                    print("Function:{0}".format(str(function.__name__)))
                    print("     Input(s):{0}".format(str(args)))
                    print("     Ouput(s):{0}".format(str(funcResult)))
                return funcResult
            except Exception as e:
                print(
                        "{0} - function failed -|- Function arguments were:{1}.".format(str(function.__name__),
                                                                                        str(args)))
                print(e.args[0])

        return funcWrapper

    if not function:  # User passed in a bool argument
        def waiting_for_function(function):
            return funcReport_Decorator(function)

        return waiting_for_function
    else:
        return funcReport_Decorator(function)


def arcToolReport(function=None, arcToolMessageBool=False, arcProgressorBool=False):
    """This decorator function is designed to be used as a wrapper with other GIS functions to enable basic try and except
     reporting (if function fails it will report the name of the function that failed and its arguments. If a report
      boolean is true the function will report inputs and outputs of a function.-David Wasserman"""

    def arcToolReport_Decorator(function):
        def funcWrapper(*args, **kwargs):
            try:
                funcResult = function(*args, **kwargs)
                if arcToolMessageBool:
                    arcpy.AddMessage("Function:{0}".format(str(function.__name__)))
                    arcpy.AddMessage("     Input(s):{0}".format(str(args)))
                    arcpy.AddMessage("     Ouput(s):{0}".format(str(funcResult)))
                if arcProgressorBool:
                    arcpy.SetProgressorLabel("Function:{0}".format(str(function.__name__)))
                    arcpy.SetProgressorLabel("     Input(s):{0}".format(str(args)))
                    arcpy.SetProgressorLabel("     Ouput(s):{0}".format(str(funcResult)))
                return funcResult
            except Exception as e:
                arcpy.AddMessage(
                        "{0} - function failed -|- Function arguments were:{1}.".format(str(function.__name__),
                                                                                        str(args)))
                print(
                        "{0} - function failed -|- Function arguments were:{1}.".format(str(function.__name__),
                                                                                        str(args)))
                print(e.args[0])

        return funcWrapper

    if not function:  # User passed in a bool argument
        def waiting_for_function(function):
            return arcToolReport_Decorator(function)

        return waiting_for_function
    else:
        return arcToolReport_Decorator(function)


def getFields(featureClass, excludedTolkens=["OID", "Geometry"], excludedFields=["shape_area", "shape_length"]):
    try:
        fcName = os.path.split(featureClass)[1]
        field_list = [f.name for f in arcpy.ListFields(featureClass) if f.type not in excludedTolkens
                      and f.name.lower() not in excludedFields]
        arcPrint("The field list for {0} is:{1}".format(str(fcName), str(field_list)), True)
        return field_list
    except:
        arcPrint("Could not get fields for the following input {0}, returned an empty list.".format(str(featureClass)),
                 True)
        arcpy.AddWarning(
                "Could not get fields for the following input {0}, returned an empty list.".format(str(featureClass)))
        field_list = []
        return field_list


@arcToolReport
def FieldExist(featureclass, fieldname):
    """ Check if a field in a feature class field exists and return true it does, false if not."""
    fieldList = arcpy.ListFields(featureclass, fieldname)
    fieldCount = len(fieldList)
    if (fieldCount >= 1) and fieldname.strip():  # If there is one or more of this field return true
        return True
    else:
        return False


@arcToolReport
def AddNewField(in_table, field_name, field_type, field_precision="#", field_scale="#", field_length="#",
                field_alias="#", field_is_nullable="#", field_is_required="#", field_domain="#"):
    # Add a new field if it currently does not exist...add field alone is slower than checking first.
    if FieldExist(in_table, field_name):
        print(field_name + " Exists")
        arcpy.AddMessage(field_name + " Exists")
    else:
        print("Adding " + field_name)
        arcpy.AddMessage("Adding " + field_name)
        arcpy.AddField_management(in_table, field_name, field_type, field_precision, field_scale,
                                  field_length,
                                  field_alias,
                                  field_is_nullable, field_is_required, field_domain)


@arcToolReport
def arcPrint(string, progressor_Bool=False):
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


# Function Definitions
def classify_features_meanshift(in_fc, search_radius, bin_seeding=False, min_bin_freq=1, cluster_all_pts=True,
                                estimate_bandwidth=False):
    """Take in a feature class of points and classify them into clusters using Mean Shift clustering from Scikit learn.
     Append field labels to the input feature class using Extend Numpy Array function."""
    try:
        # Declare Starting Variables
        desc = arcpy.Describe(in_fc)
        OIDFieldName = desc.OIDFieldName
        workspace = os.path.dirname(desc.catalogPath)
        arcPrint("Converting '{0}' feature class geometry to X-Y centroid numpy arrays.".format(str(desc.name)))
        centroid = 'SHAPE@XY'
        objectid = 'OID@'
        fields = [centroid, objectid]
        # Convert Feature Class to NP array
        geoarray = arcpy.da.FeatureClassToNumPyArray(in_fc, fields,
                                                     null_value=1)  # Null Values of treated as one feature -weight
        coordinates_cluster = geoarray[centroid]
        if estimate_bandwidth:
            search_radius = cluster.estimate_bandwidth(coordinates_cluster)
            arcPrint("Using estimated bandwidth of {0} based on estimation function.".format(search_radius), True)
        arcPrint("Using geographic coordinates to classify with Mean_Shift.", True)
        meanshift_classification = cluster.MeanShift(bandwidth=search_radius, bin_seeding=bin_seeding,
                                                     min_bin_freq=min_bin_freq, cluster_all=cluster_all_pts).fit(
                coordinates_cluster)
        cluster_centroids = meanshift_classification.cluster_centers_
        labels = meanshift_classification.labels_
        # Number of clusters in labels, ignoring noise if present.
        cluster_count = len(set([i for i in labels if i != -1]))
        arcPrint('Estimated number of clusters: {0}'.format(cluster_count), True)
        try:
            arcPrint("Silhouette Coefficient: {0}.".format(metrics.silhouette_score(coordinates_cluster, labels)), True)
            arcPrint(
                    """Wikipedia: The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.""")
        except Exception as e:
            arcPrint("Could not compute Silhouette Coefficient. Error: {0}".format(str(e.args[0])), True)
        arcPrint("Appending Labels from Mean Shift to new numpy array.", True)
        JoinField = str(arcpy.ValidateFieldName("NPIndexJoin", workspace))
        LabelField = str(arcpy.ValidateFieldName("MeanShiftLabel", workspace))
        finalMean_ShiftArray = np.array(list(zip(geoarray[objectid], labels)),
                                        dtype=[(JoinField, np.int32), (LabelField, np.int32)])
        arcPrint("Extending Label Fields to Output Feature Class. Clusters labels start at 0, noise is labeled -1.",
                 True)
        arcpy.da.ExtendTable(in_fc, OIDFieldName, finalMean_ShiftArray, JoinField, append_only=False)
        del geoarray, finalMean_ShiftArray, labels, meanshift_classification
        arcPrint("Script Completed Successfully.", True)
    except arcpy.ExecuteError:
        arcPrint(arcpy.GetMessages(2))
    except Exception as e:
        arcPrint(e.args[0])


# End do_analysis function

# This test allows the script to be used from the operating
# system command prompt (stand-alone), in a Python IDE,
# as a geoprocessing script tool, or as a module imported in
# another script
if __name__ == '__main__':
    classify_features_meanshift(input_feature_class, search_radius=bandwidth,
                                estimate_bandwidth=use_estimated_bandwidth, cluster_all_pts=cluster_all_points)
