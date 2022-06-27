cd ../../External_data/ENTSO-E

# To access the sftp you need an account for the ENTSO-E Transparency platform
sftp INSERT_EMAIL@sftp-transparency.entsoe.eu << EOF

cd TP_export

# Actual load
cd ActualTotalLoad_6.1.A
get 201[5-9]*.csv

# Load forecast
cd ..
cd DayAheadTotalLoadForecast_6.1.B
get 201[5-9]*.csv

# Generation forecast
cd ..
cd DayAheadAggregatedGeneration_14.1.C
get 201[5-9]*.csv

# Generation per type
cd ..
cd AggregatedGenerationPerType_16.1.B_C
get 201[5-9]*.csv

# Wind/solar forecast
cd ..
cd DayAheadGenerationForecastForWindAndSolar_14.1.D
get 201[5-9]*.csv

# Day ahead prices
cd ..
cd DayAheadPrices_12.1.D
get 201[5-9]*.csv

#Commercial Schedules
cd ..
cd TotalCommercialSchedules_12.1.F
get 201[5-9]*.csv

#Physical Flows
cd ..
cd PhysicalFlows_12.1.G
get 201[5-9]*.csv

#Generation per unit
cd ..
cd ActualGenerationOutputPerGenerationUnit_16.1.A
get 201[5-9]*.csv

Outages
cd ..
cd UnavailabilityInTransmissionGrid_10.1.A_B
get 201[5-9]*.csv



EOF
