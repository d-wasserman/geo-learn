# --------------------------------
# Name: GeoDBSCAN.py
# Purpose: This script is intended to allow ArcGIS users that have Scikit Learn installed in their python installation
# utilize DBSCAN to create clusters of geographic features based on their centroids.
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
import os, arcpy
import numpy as np
import pandas as pd
import glearnlib as gl

try:
    from sklearn import cluster
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
except:
    arcpy.AddError("This library requires Sci-kit Learn installed in the ArcGIS Python Install."
                   " Might require installing pre-requisite libraries and software.")


# Function Definitions
def classify_features_dbscan(in_fc, neighborhood_size, minimum_samples, weight_field):
    """Take in a feature class of points and classify them into clusters using DBSCAN from Scikit learn.
     Append field labels to the input feature class using Extend Numpy Array function."""
    try:
        # Declare Starting Variables
        desc = arcpy.Describe(in_fc)
        OIDFieldName = desc.OIDFieldName
        workspace = os.path.dirname(desc.catalogPath)
        gl.arc_print("Converting '{0}' feature class geometry to X-Y centroid numpy arrays.".format(str(desc.name)))
        centroid_x, centroid_y = 'SHAPE@X', 'SHAPE@Y'
        objectid = 'OID@'
        fields = [centroid_x, centroid_y, objectid]
        use_weight = False
        if gl.field_exist(in_fc, weight_field):
            fields.append(weight_field)
            use_weight = True
        # Convert Feature Class to NP array
        geoarray = arcpy.da.FeatureClassToNumPyArray(in_fc, fields,
                                                     null_value=1)  # Null Values of treated as one feature -weight
        cluster_fields = [centroid_x, centroid_y]
        data = pd.DataFrame(geoarray)
        coordinates_cluster = data[cluster_fields]
        if use_weight:
            gl.arc_print("Using weight field {0} and geographic coordinates for clustering with DBSCAN.".format(
                str(weight_field)), True)
            weight = np.asarray(data[weight_field], dtype=np.float64)
            dbscan_classification = cluster.DBSCAN(neighborhood_size, minimum_samples).fit(coordinates_cluster, weight)
        else:
            gl.arc_print("Using geographic coordinates to classify with DBSCAN.", True)
            dbscan_classification = cluster.DBSCAN(neighborhood_size, minimum_samples).fit(coordinates_cluster)
        core_samples_mask = np.zeros_like(dbscan_classification.labels_, dtype=bool)
        core_samples_mask[dbscan_classification.core_sample_indices_] = True
        labels = dbscan_classification.labels_
        # Number of clusters in labels, ignoring noise if present.
        cluster_count = len(set([i for i in labels if i != -1]))
        gl.arc_print('Estimated number of clusters: {0}'.format(cluster_count), True)
        try:
            gl.arc_print("Silhouette Coefficient: {0}.".format(metrics.silhouette_score(coordinates_cluster, labels)),
                         True)
            gl.arc_print(
                """Wikipedia: The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.""")
        except Exception as e:
            gl.arc_print("Could not compute Silhouette Coefficient. Error: {0}".format(str(e.args[0])), True)
        gl.arc_print("Appending Labels from DBSCAN to new numpy array.", True)
        JoinField = str(arcpy.ValidateFieldName("NPIndexJoin", workspace))
        LabelField = str(arcpy.ValidateFieldName("DBSCANLabel", workspace))
        finalDBSCANArray = np.array(list(zip(data[objectid], labels)),
                                    dtype=[(JoinField, np.int32), (LabelField, np.int32)])
        gl.arc_print("Extending Label Fields to Output Feature Class. Clusters labels start at 0, noise is labeled -1.",
                     True)
        arcpy.da.ExtendTable(in_fc, OIDFieldName, finalDBSCANArray, JoinField, append_only=False)
        del geoarray, finalDBSCANArray, labels, dbscan_classification, core_samples_mask
        gl.arc_print("Script Completed Successfully.", True)
    except arcpy.ExecuteError:
        gl.arc_print(arcpy.GetMessages(2))
    except Exception as e:
        print(str(e.args[0]))
        arcpy.AddError(str(e.args[0]))


# End do_analysis function

# This test allows the script to be used from the operating
# system command prompt (stand-alone), in a Python IDE,
# as a geoprocessing script tool, or as a module imported in
# another script
if __name__ == '__main__':
    # Define input parameters

    input_feature_class = arcpy.GetParameterAsText(0)
    neighborhood_size = arcpy.GetParameter(1)
    minimum_samples = arcpy.GetParameter(2)
    weight_field = arcpy.GetParameterAsText(3)
    classify_features_dbscan(input_feature_class, neighborhood_size, minimum_samples, weight_field)
