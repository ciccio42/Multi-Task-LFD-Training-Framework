# train_flow

## generate command embds
* use USE (universal sentence encoder) to generate embeddings
`bashes/command_encoder.sh`

* the command are saved in the following directory:
` path/to/commands `

## train cond 
* train CondModule on embeddings (their centroids) generated by the USE
`bashes/train_cond_module.sh`

## train RT-1
* train RT-1 with CondModule freezed