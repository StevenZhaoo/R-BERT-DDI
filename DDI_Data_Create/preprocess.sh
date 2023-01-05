# Replace space to underscore in filenames

export XML_DIR=DDIdata
export BRAT_DIR=DDIbrat
export TSV_FILE=result.txt


for i in $XML_DIR/*.xml; do
    mv "$i" `echo $i | sed -e 's/ /_/g'`
done

# Convert XML to Brat format
# mkdir $BRAT_DIR
for i in $XML_DIR/*.xml; do
    python xml2brat.py $i $BRAT_DIR/`basename $i .xml`
done

# Convert Brat to TSV format
# mkdir $TSV_DIR
 
touch $TSV_FILE

# python3 brat2tsv.py $BRAT_DIR $TSV_DIR/$TSV_NAME
python brat2tsv.py $BRAT_DIR $TSV_FILE
