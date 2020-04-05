# --------------------------------
# Name: GeoMeanShift.py
# Purpose: This script is intended to allow ArcGIS users that have Scikit Learn installed in their python installation
# utilize Mean Shift to create clusters of geographic features based on their centroids.
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
def classify_features_meanshift(in_fc, search_radius, output_fc, weight_field=None, alternative_fields=[],
                                bin_seeding=False, min_bin_freq=1, cluster_all_pts=True, estimate_bandwidth=False):
    """Take in a feature class of points and classify them into clusters using Mean Shift clustering from Scikit learn.
     Append field labels to the input feature class using Extend Numpy Array function."""
    try:
        # Declare Starting Variables
        arcpy.env.overwriteOutput = True
        desc = arcpy.Describe(in_fc)
        SpatialReference = desc.spatialReference
        workspace = os.path.dirname(desc.catalogPath)
        gl.arc_print("Converting '{0}' feature class to numpy array based on inputs.".format(str(desc.name)))
        centroid_x, centroid_y = 'SHAPE@X', 'SHAPE@Y'
        OIDFieldName = desc.OIDFieldName
        feature_class_fields, cluster_fields = gl.determine_extract_and_subset_fields(in_fc,
                                                                                      alternative_fields,
                                                                                      [centroid_x, centroid_y],
                                                                                      [OIDFieldName, weight_field],
                                                                                      [weight_field])
        gl.arc_print("Feature class clustering will be conducted on the following fields: {0}".format(cluster_fields))
        # Convert Feature Class to NP array
        geoarray = arcpy.da.FeatureClassToNumPyArray(in_fc, feature_class_fields,
                                                     null_value=1)  # Null Values of treated as one feature -weight
        data = pd.DataFrame(geoarray[cluster_fields])
        # Create Weighted arrays if weight field is present.
        using_cluster_weight = True if weight_field in feature_class_fields else False
        if using_cluster_weight:
            gl.arc_print("Preparing weighted Data for clustering.")
            data = gl.return_weighted_array(data, geoarray[weight_field])
        # Standardize Data if using Fields.
        clustering_on_geometry = True if centroid_x and centroid_y in cluster_fields else False
        if not clustering_on_geometry:  # If Clustering on arbitrary fields, standardize data.
            gl.arc_print(
                "Processing arbitrary fields rather than feature coordinates. Standardizing data with Sklearn's "
                "StandardScaler(). Bandwidth should be in standardized units or using the estimated bandwidth.")
            scaler = StandardScaler().fit(data)
            data_to_cluster = scaler.transform(data)
        else:
            data_to_cluster = data
        # Estimate Bandwidth if chosen.
        if estimate_bandwidth or search_radius <= 0.0:
            search_radius = cluster.estimate_bandwidth(data_to_cluster)
            gl.arc_print("Using estimated bandwidth of {0} based on estimation function.".format(search_radius), True)
        gl.arc_print("Using geographic coordinates to classify with Mean_Shift.", True)
        meanshift_classification = cluster.MeanShift(bandwidth=search_radius, bin_seeding=bin_seeding,
                                                     min_bin_freq=min_bin_freq, cluster_all=cluster_all_pts).fit(
            data_to_cluster)
        cluster_centroids = meanshift_classification.cluster_centers_
        labels = meanshift_classification.labels_
        # Number of clusters in labels, ignoring noise if present.
        unique_clusters = set([i for i in labels if i != -1])
        cluster_count = len(unique_clusters)
        gl.arc_print('Estimated number of clusters: {0}'.format(cluster_count), True)
        try:
            gl.arc_print("Silhouette Coefficient: {0}.".format(metrics.silhouette_score(data_to_cluster, labels)), True)
            gl.arc_print(
                """Wikipedia: The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from -1 to 1, where a high value indicate that the object is well matched to its own cluster and poorly matched to neighboring clusters.""")
        except Exception as e:
            gl.arc_print("Could not compute Silhouette Coefficient. Error: {0}".format(str(e.args[0])), True)
        # After Clustering and Metric gathering extend feature class and export.
        gl.arc_print("Appending Labels from Mean Shift to new numpy array.", True)
        JoinField = str(arcpy.ValidateFieldName("NPIndexJoin", workspace))
        LabelField = str(arcpy.ValidateFieldName("MeanShiftLabel", workspace))
        LabelCount = str(arcpy.ValidateFieldName("LabelCount", workspace))
        ShapeXField = str(arcpy.ValidateFieldName("ShapeX", workspace))
        ShapeYField = str(arcpy.ValidateFieldName("ShapeY", workspace))
        finalMean_ShiftArray = np.array(list(zip(geoarray[OIDFieldName], labels)),
                                        dtype=[(JoinField, np.int32), (LabelField, np.int32)])
        gl.arc_print("Extending Label Fields to Output Feature Class. Clusters labels start at 0, noise is labeled -1.",
                     True)
        if using_cluster_weight:
            labels = gl.reduce_weighted_array(labels, geoarray[weight_field])
        arcpy.da.ExtendTable(in_fc, OIDFieldName, finalMean_ShiftArray, JoinField, append_only=False)
        # Export feature class centroids
        directory_name = os.path.split(output_fc)[0]
        file_name = os.path.split(output_fc)[1]
        if arcpy.Exists(directory_name) and clustering_on_geometry:
            # Only create new feature class it output locations exists and if there clustering is on geometry.
            gl.arc_print("Creating Centroid Feature Class of clusters {0}.".format(str(file_name)), True)
            ShapeX, ShapeY = zip(*cluster_centroids)
            count_of_items_per_label = [int(labels.tolist().count(unique_value)) for unique_value in unique_clusters]
            final_centroid_array = np.asarray(list(zip(ShapeX, ShapeY, unique_clusters, count_of_items_per_label)),
                                              dtype=[(ShapeXField, np.float64), (ShapeYField, np.float64),
                                                     (LabelField, np.int32), (LabelCount, np.int32)])
            arcpy.da.NumPyArrayToFeatureClass(final_centroid_array, output_fc, (ShapeXField, ShapeYField),
                                              SpatialReference)
        del geoarray, finalMean_ShiftArray, labels, meanshift_classification
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
    bandwidth = arcpy.GetParameter(1)
    use_estimated_bandwidth = arcpy.GetParameter(2)
    cluster_all_points = arcpy.GetParameter(3)
    output_feature_class_centroids = arcpy.GetParameterAsText(4)
    weight_field = arcpy.GetParameterAsText(5)
    fields_to_cluster = str(arcpy.GetParameterAsText(6)).split(";")
    classify_features_meanshift(input_feature_class, weight_field=weight_field,
                                search_radius=bandwidth, output_fc=output_feature_class_centroids,
                                estimate_bandwidth=use_estimated_bandwidth, cluster_all_pts=cluster_all_points)
