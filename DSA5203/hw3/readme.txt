To run training, you might need adjust the batch size in the train function directly if you run into OOM (out of memory) error as we used batch_size=256 for training, which may be too large for some GPU RAM.

Same goes for testing function. You might need to adjust batch_size accordingly, but we used 32 for test so should be okay. Thanks!
Please contact e0540498@u.nus.edu for clarifications!