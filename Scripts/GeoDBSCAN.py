# --------------------------------
# Name: GeoDBSCAN.py
# Purpose: This script is intended to allow ArcGIS users that have Scikit Learn installed in their python installation
# utilize DBSCAN to create clusters of geographic features based on their centroids.
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
neighborhood_size = arcpy.GetParameter(1)
minimum_samples = arcpy.GetParameter(2)
weight_field = arcpy.GetParameterAsText(3)


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

# Function Definitions
def classify_features_dbscan(in_fc, neighborhood_size, minimum_samples, weight_field):
    """Take in a feature class of points and classify them into clusters using DBSCAN from Scikit learn.
     Append field labels to the input feature class using Extend Numpy Array function."""
    try:
        # Declare Starting Variables
        desc = arcpy.Describe(in_fc)
        OIDFieldName=desc.OIDFieldName
        workspace= os.path.dirname(desc.catalogPath)
        arc_print("Converting '{0}' feature class geometry to X-Y centroid numpy arrays.".format(str(desc.name)))
        centroid_x,centroid_y = 'SHAPE@X','SHAPE@Y'
        objectid = 'OID@'
        fields = [centroid_x,centroid_y, objectid]
        use_weight = False
        if field_exist(in_fc, weight_field):
            fields.append(weight_field)
            use_weight = True
        # Convert Feature Class to NP array
        geoarray = arcpy.da.FeatureClassToNumPyArray(in_fc, fields,
                                                     null_value=1)  # Null Values of treated as one feature -weight
        cluster_fields=[centroid_x,centroid_y]
        coordinates_cluster = geoarray[cluster_fields].view((np.float64,len(cluster_fields)))

        if use_weight:
            arc_print("Using weight field {0} and geographic coordinates for clustering with DBSCAN.".format(
                        str(weight_field)),True)
            weight = np.asarray(geoarray[weight_field], dtype=np.float64)
            dbscan_classification = cluster.DBSCAN(neighborhood_size, minimum_samples).fit(coordinates_cluster, weight)
        else:
            arc_print("Using geographic coordinates to classify with DBSCAN.",True)
            dbscan_classification = cluster.DBSCAN(neighborhood_size, minimum_samples).fit(coordinates_cluster)
        core_samples_mask = np.zeros_like(dbscan_classification.labels_, dtype=bool)
        core_samples_mask[dbscan_classification.core_sample_indices_] = True
        labels = dbscan_classification.labels_
        # Number of clusters in labels, ignoring noise if present.
        cluster_count = len(set([i for i in labels if i!=-1]))
        arc_print('Estimated number of clusters: {0}'.format(cluster_count),True)
        try:
            arc_print("Silhouette Coefficient: {0}.".format(metrics.silhouette_score(coordinates_cluster, labels)),True)
            arc_print("""Wikipedia: The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.""")
        except Exception as e:
            arc_print("Could not compute Silhouette Coefficient. Error: {0}".format(str(e.args[0])),True)
        arc_print("Appending Labels from DBSCAN to new numpy array.",True)
        JoinField=str(arcpy.ValidateFieldName("NPIndexJoin",workspace))
        LabelField= str(arcpy.ValidateFieldName("DBSCANLabel",workspace))
        finalDBSCANArray= np.array(list(zip(geoarray[objectid],labels)),dtype=[(JoinField,np.int32),(LabelField,np.int32)])
        arc_print("Extending Label Fields to Output Feature Class. Clusters labels start at 0, noise is labeled -1.",True)
        arcpy.da.ExtendTable(in_fc,OIDFieldName,finalDBSCANArray,JoinField,append_only=False)
        del geoarray, finalDBSCANArray,labels, dbscan_classification, core_samples_mask
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
    classify_features_dbscan(input_feature_class, neighborhood_size, minimum_samples, weight_field)
