# Instruction to PrivacyRestore

## Meanings of Each Subdir
1. ./Pri-DDXPlus: The code for PrivacyRestore on Pri-DDXPlus dataset
2. ./Pri-NLICE: The code for PrivacyRestore on Pri-NLICE dataset
3. ./Pri-SLJA: The code for PrivacyRestore on Pri-SLJA dataset

## The file structure of PrivacyRestore code
- ./llama/ : contain the base Llama model structure and definition.
- ./BankConstruction: contain the code for building Restore Bank.
- ./Training: contain the code for Restoraion Vectors Training and evaluation code.
- ./gpt_utils: contain the code for computing LLM-J score
- ./dx_privacy contain the code for applying $d_\chi$-privacy


## PrivacyRestore

### [STEP1] Prepare dataset and change the working directory
```bash
cd Pri-DDXPlus
```

### [STEP1] Prepare for PrivacyRestore
```bash
# init the restoraion vectors and obtain the common top-K heads set.
1. cd ./sh
2. fill the parameters in prepare.sh
3. sh prepare.sh
``` 

```bash
# train the restoration vectors
1. fill the parameters in the train.sh.
2. sh train.sh
```

### [STEP2] Evaluate PrivacyRestore
```bash
# compute mc1 and mc2 metric
1. fill the parameters in the test.sh.
2. sh test.sh
```