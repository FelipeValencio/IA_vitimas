import csv

# Open the input and output files
with open('resultsRFReg.txt', 'r') as infile, open('output2.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)

    # Write the header row to the output file
    writer.writerow(['MSE', 'RMSE', 'MAE'])

    # Loop over the lines in the input file
    for line in infile:
        # Strip the newline character from the end of the line
        line = line.strip()

        # Split the line into three parts, based on the commas
        parts = line.split(',')

        # Extract the values for MSE, RMSE, and MAE from the parts
        mse = float(parts[0].split(':')[1].strip())
        rmse = float(parts[1].split(':')[1].strip())
        mae = float(parts[2].split(':')[1].strip())

        # Write a row to the output file with the values for MSE, RMSE, and MAE
        writer.writerow([mse, rmse, mae])
