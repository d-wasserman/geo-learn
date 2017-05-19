# Distribution Summary

This is a collection scripts and tools intended to provide a template on how to integrate and apply Scikit-Learn with ArcGIS Pro. The tools distributed enable access to various machine learning through scripting tools in the GeoLearn toolbox. The tools largely work by passing geographic coordinates and related data to be clustered or analyzed to help with spatial analysis tasks, data reduction, or cartography.  

# Usage

All tools have PDF help documents that describe the algorithms used and the parameters used to use the scripting tools. The help documents also have links to the sci-kit learn documentation and other relevant pages to understand how the various algorithms work. 

For more information about Scikit-Learn check out their main landing page here: http://scikit-learn.org/

# Installing Scikit-Learn for ArcGIS Pro

The general steps for installing Scikit-Learn is the same for installing other 3rd Party Libraries. 

:globe_with_meridians: Navigate to the python folder with the ArcGIS Pro install. Usually follows in Program Files-->/ArcGIS\Pro\bin\Python\envs\arcgispro-py3. Activate your conda environment. Newer versions of pro might make this unnecessary.  

:globe_with_meridians: Open Command Prompt or your connected IDE (might need to run as adminstrator). Newer versions of pro might make this unnecessary.

:globe_with_meridians: Add Packages with your IDE, Command Prompt, or with ArcGIS Pros Python Package Installer. Conda command is conda install scikitlearn. 

:globe_with_meridians: ArcGIS Pros Python Package Manager is a quick and easy way to add the required dependencies: http://pro.arcgis.com/en/pro-app/arcpy/get-started/what-is-conda.htm


Relevant Links:

1.https://geonet.esri.com/docs/DOC-8359

2.http://conda.pydata.org/docs/using/pkgs.html#install-a-package

3.https://docs.continuum.io/anaconda/ide_integration

4.http://scikit-learn.org/stable/install.html

# Typical Parameters

Instead of documenting each tool, the table below documents general parameters these scripting tools have. More advance analysis will require script and parameter customization.

<table width="100%" border="0" cellpadding="5">
<tbody>
<tr>
<th width="30%">
<b>Parameter</b>
</th>
<th width="50%">
<b>Explanation</b>
</th>
<th width="20%">
<b>Data Type</b>
</th>
</tr>
<tr>
<td class="info">Input_Feature_Class</td>
<td class="info" align="left">
<span style="font-weight: bold">Dialog Reference</span><br /><DIV STYLE="text-align:Left;"><DIV><P><SPAN>This is the input feature class that will be clustered or fit using the chosen tool algorithm and its labels/predicted values will be added to the feature class fields. In the case of regression based analysis tools, output summary files might be elected for. </SPAN></P><P><SPAN /></P></DIV></DIV><p><span class="noContent"></span></p></td>
<td class="info" align="left">Feature Layer</td>
</tr>
<tr>
<td class="info">Neighborhood Size/Band Width/Cluster Count/Sensitivity Parameters</td>
<td class="info" align="left">
<span style="font-weight: bold">Dialog Reference</span><br /><DIV STYLE="text-align:Left;"><DIV><DIV><P><SPAN> Various clustering/regression algorithms have a bandwidth/alpha/k clusters that specify the parameters of the fit. There will be parameters exposed in the tool to specify these parameters. In the case of bandwidth they will be in the units of the current projection.</SPAN></P><P><SPAN>The points are represented by the raw centroid coordinates returned by "SHAPE@XY" token. </SPAN></P></DIV></DIV></DIV><p><span class="noContent"></span></p></td>
<td class="info" align="left">Double</td>
</tr>
<tr>
<td class="info">Weight Fields or Variable Selection(Optional) </td>
<td class="info" align="left">
<span style="font-weight: bold">Dialog Reference</span><br /><DIV STYLE="text-align:Left;"><DIV><P><SPAN> For clustering this could be a field used to weight the points according to some magnitude that point represents (one point could represent 5 entities), or there will be parameters to select independent and dependent variables for regression analysis. </SPAN></P></DIV></DIV><p><span class="noContent"></span></p></td>
<td class="info" align="left">Field</td>
</tr>
</tbody>
</table>
