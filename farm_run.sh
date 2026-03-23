#### Assumption: we are running in ~/scratch/OptimalSGT/

HOME_DIR="$HOME/scratch/SagiXGBoostTreeApproximator"
FARM1="$HOME_DIR/farm1" # ~/scratch/OptimalSGT/farm1/table.dat
FARM1_TABLE="farm1/table.dat"


### FARM 1 SETUP ### 
FARM1_COUNT=0
# clear everything in FARM1_TABLE
> $FARM1_TABLE



for FILE in xgb_run_new.py shapefbt_run_new.py
do
    for dataset in room avila bank bean bidding eye-state fault htru magic occupancy page raisin rice segment skin wilt
    do
        for i in $(seq 1 200)
        do
            for fold in 0 1 2 3 4
            do
                # write to FARM1_TABLE
                ((FARM1_COUNT++))
                echo "$FARM1_COUNT python $HOME_DIR/tests/$FILE --dataset $dataset --fold $fold --trial-id $i --home-dir $HOME_DIR" >> $FARM1_TABLE 
            done
        done
    done
done




# remove any trailing empty lines from FARM1_TABLE
sed -i -e :a -e '/^\s*$/d;N;ba' $FARM1_TABLE
echo "Total FARM1 jobs: $FARM1_COUNT"
