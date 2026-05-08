# Healthcare Infrastructure EDA Report

## Hospital Directory Analysis
- **Total Facilities**: 30273
- **Top State**: Maharashtra
- **Ownership**: The majority are 0.

## Rural Health Analysis (Focus: Tiruppur)
- **Total Facilities**: 55
- **Accessibility**: Significant presence of PRIVATE facilities.

## Recommendation System Readiness
- **Geo-info**: High availability of Lat/Long for rural facilities, but sparse for national directory.
- **Specialties**: Data on specialized services is present but requires significant normalization.
- **Feasibility**: High. We can recommend facilities based on state/district and proximity where geo-data is available.

## Recommendations
- Standardize 'specialties' list using a medical taxonomy.
- Impute missing geo-coordinates using city/district names if possible.
- Merge rural and national directories into a unified recommendation graph.
