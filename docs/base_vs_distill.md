# Base (100 NFE) vs Distilled (8 NFE)

[← Back to README](../README.md).

## Run Base and Distilled Model

```bash
# Taking T2I for example
# Run Base
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 4.0 --cfg_norm none --timestep_shift 3.0 --num_steps 50 \
    --profile


# Run 8-step preview model (deprecated)
python examples/t2i/inference.py \
    --model_path SenseNova-U1-8B-MoT-8step-preview \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 --num_steps 8 \
    --profile

# Run 8-step LoRA
huggingface-cli download sensenova/SenseNova-U1-8B-MoT-LoRAs --include "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors" --local-dir ./sensenova/SenseNova-U1-8B-MoT-LoRAs/ --local-dir-use-symlinks False
python examples/t2i/inference.py \
    --model_path sensenova/SenseNova-U1-8B-MoT \
    --lora_path sensenova/SenseNova-U1-8B-MoT-LoRAs/SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors \
    --jsonl examples/t2i/data/samples.jsonl \
    --output_dir outputs/ \
    --cfg_scale 1.0 --cfg_norm none --timestep_shift 3.0 --num_steps 8 \
    --profile
```

---

## Text-to-Image


| SenseNova-U1-8B-MoT (100 NFE) | SenseNova-U1-8B-MoT-8step-preview (8 NFE) |
|---|---|
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/b957b661-f916-40b7-8457-bb76ed942ce1" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/5f921347-abc7-4034-b37e-166150c46116" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/2b06c91d-320b-4344-b2b7-3eecc3655afb" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/aacbf7cd-6641-4046-a148-e56227fc1ba6" /> |
| <img width="2720" height="1536" alt="Image" src="https://github.com/user-attachments/assets/eeec15de-6004-4acf-9947-b29cea0cd404" /> | <img width="2720" height="1536" alt="Image" src="https://github.com/user-attachments/assets/2c6a933f-93bb-4b28-a959-9998187626be" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/ebfa083c-e1a3-47b1-baec-18204c19ab8a" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/11326e6c-134a-4ceb-ac17-07ddcc430200" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/aabfeca6-f346-4787-9bbf-b90c66250262" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/ae14ae4f-ba42-4c49-8d29-d76031dff130" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/c60dc057-301f-4df4-ae73-270a67ea928f" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/93b76c40-3333-4738-b977-7981635b162b" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/1d90cfd5-54ae-4669-9cde-51b0b911405e" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/52db191b-dce3-4c85-bfb7-e02a45bef0fa" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/eaf8f018-de5d-4ae6-bfde-895c12833480" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/2a1f04c7-a990-403c-904f-28aaa31f1c33" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/fc7d04f5-f2f0-4aa0-8cda-f0f06defcdd9" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/e96c4e91-2d99-42dc-a521-8a8922fbf580" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/3a949ced-c7f1-4175-b478-60792fd634a4" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/cbcda1d3-ec83-4d3a-b1b5-1130fc67095e" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/dc93499e-f7f7-4d08-8d6f-69f27b3b943e" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/fd868c8e-ddaf-4fa0-b3dc-976cab223443" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/e62aa2ed-650a-44b5-b74d-b8aa150ec8e5" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/7b7bc125-5d56-43f2-9938-a9c73489a563" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/8704123c-7afe-4a77-a878-883c354d9e27" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/dfea87b5-0295-4713-8875-313d041a1623" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/c79cd8dc-53c7-4586-b7d0-9002db1b9829" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/e91b1502-a93f-475f-b408-141748e57af5" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/d158e591-fa2e-4e5d-b61c-1e5f4692b529" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/158ebf30-144d-4cb9-8708-5eca39def04e" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/4f92e25a-3d4f-4b0f-9e13-9027dc688df6" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/0e1104f3-a5df-4291-bb34-5ecd847f12bb" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/c11fedcb-8362-4315-8cf6-7a6652746260" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/644938ee-adbf-4f2e-beec-9ace2abb7a3e" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/79982f47-76d0-42fd-bac2-783cdc3c6cab" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/54562cd5-895c-4ee1-a4da-371865692eae" /> |
| <img width="2496" height="1664" alt="Image" src="https://github.com/user-attachments/assets/ca7519ee-bfbc-41d3-acb7-070cbf8fc588" /> | <img width="2496" height="1664" alt="Image" src="https://github.com/user-attachments/assets/60ff8170-5826-4ffc-97b4-9324d58b98cb" /> |
| <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/16b69220-2594-4517-beca-f70954895937" /> | <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/0af69452-5e1d-4982-a26b-a860c7051656" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/46c13011-9f86-4ed1-b167-0e0112606136" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/021349a3-d92a-4d3d-930d-50d3afa5adc1" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/920a325a-ba53-4dfd-8843-863cc20a2d44" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/8d82ee0f-2051-4112-a47e-31ed03fce4d5" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/08232762-81fb-4aa6-b001-389064e73d2e" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/ec84c507-8c79-4a74-b325-3ba1e5196f69" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/ecebe0ae-9f83-430b-ad0d-1c26689cadc8" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/ab892f47-6786-4c91-a736-0a034e3e8b3c" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/da932a5c-e13d-43f6-a619-6c057a7d4826" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/8eb5874e-a8fe-427a-a5f0-8633681a77ec" /> |
| <img width="1664" height="2496" alt="Image" src="https://github.com/user-attachments/assets/522c9f71-56c0-4924-b099-61bc19646a7e" /> | <img width="1664" height="2496" alt="Image" src="https://github.com/user-attachments/assets/7fdd668f-d6d6-4295-91c8-f3f2caa43bd6" /> |
| <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/996f59cc-eabb-45c1-b9b9-f2553fc42ec8" /> | <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/07cfc906-ac47-496e-97b5-19c5ff8f785a" /> |
| <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/bcf618f1-50fe-40b7-8d14-a14bc294abf3" /> | <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/b2e3c2a0-2970-40f1-b678-5af71d7640c9" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/018504fd-ffc4-44ac-b8d0-ff2109cd4b2e" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/bad8ac39-e41a-4e6b-a390-0b2fd99d9af7" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/28bc226b-122e-48f9-b2cd-46816274d74c" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/1390480b-f405-4615-b3b2-b0ebda9408b1" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/d0a1e419-294b-4c59-896a-a8ae85668e40" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/9d96b38a-f5e9-4c00-937f-88e6c50e50f3" /> |
| <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/45eb0bc6-bce2-42ac-b5c3-cab26663fd7d" /> | <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/ea842ff6-a621-41f5-8de1-889b0565e155" /> |
| <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/39bc2e0d-7eb7-4b94-82f9-d51766c4dc53" /> | <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/7c390a71-cce2-4c1d-bf72-c7fb73d52cec" /> |
| <img width="2720" height="1536" alt="Image" src="https://github.com/user-attachments/assets/6540e91d-012b-4d9a-b5b2-133127019a89" /> | <img width="2720" height="1536" alt="Image" src="https://github.com/user-attachments/assets/63a55bd7-95ce-4447-b184-d7bafa0bed12" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/c3f26e79-18ca-4e80-9f8a-284cff18bbbe" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/98afbfa3-4e2e-4087-8e65-cff2efd33f52" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/9675fe23-dcb0-4959-964e-bb57d71a8901" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/a1b473eb-205f-49e8-b069-8b16556c3d5a" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/a3c91778-740f-42be-9d85-cfaa2b3a34d6" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/e07f3884-d46c-4eef-af88-0b12638f7525" /> |
| <img width="1664" height="2496" alt="Image" src="https://github.com/user-attachments/assets/f76c6220-9f95-4a58-8832-ffd825f28ee9" /> | <img width="1664" height="2496" alt="Image" src="https://github.com/user-attachments/assets/d837428c-6d27-4d6a-90c4-2c29da250785" /> |
| <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/7ab2b3e8-5084-4584-9bf4-00c59ccddbf5" /> | <img width="2368" height="1760" alt="Image" src="https://github.com/user-attachments/assets/0de36a7f-28f4-40d6-9df8-a746092dbce6" /> |
| <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/a6818c15-ee27-4d0b-9e57-29ae0fe89bf9" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/c40f4df8-99a6-4ca7-b953-e0f849d6bcd7" /> |
| <img width="1760" height="2368" alt="Image" src="https://github.com/user-attachments/assets/42aa2e76-fbba-41ce-a84b-55b7e2e20f46" /> | <img width="1760" height="2368" alt="Image" src="https://github.com/user-attachments/assets/7b66e1a2-281d-4ccc-b885-872be4afa59a" /> |
| <img width="1664" height="2496" alt="Image" src="https://github.com/user-attachments/assets/f1f76b33-5eed-4e8b-9e81-27eceb61f600" /> | <img width="1664" height="2496" alt="Image" src="https://github.com/user-attachments/assets/394873c1-745f-4bb7-b1d3-e0dface04522" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/6ddcaf07-8e87-4460-b151-a596064af0e5" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/43085347-61ad-4640-9c94-903a48d0937c" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/dda19b41-5467-4375-9079-f28fec74a8d8" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/c2e3ef48-831d-446b-946b-b9c8cab81bf7" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/cb88be1c-6e4b-4e2b-924c-99d2f4ea9463" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/97900093-98ec-495c-8867-359f8d73dadd" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/6f7acfca-aeb7-4dc0-8efa-53a325f830aa" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/f2aef2a2-29b4-411d-8321-42f57662015c" /> |
| <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/f027e3d5-a08f-48bb-b2e3-9ab13b05dea2" /> | <img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/5ab3f0a0-b803-4251-b96f-0d7140ad2df1" /> |


## Image-Editing

| Reference Image | SenseNova-U1-8B-MoT (100 NFE) | SenseNova-U1-8B-MoT-8step-preview (8 NFE) |
|---|---|---|
| <img width="640" height="640" alt="3.webp" src="../examples/editing/data/images/3.webp" /> | <img width="1536" height="2752" alt="Image" src="https://github.com/user-attachments/assets/8fcbcf8f-ad6a-4ea8-af82-5c99f37ffa11" /> | <img width="1536" height="2752" alt="Image" src="https://github.com/user-attachments/assets/e6878663-e64f-407c-9f1e-88987591b977" /> |
| <img width="640" height="640" alt="6.webp" src="../examples/editing/data/images/6.webp" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/3f2341f0-b20d-44b3-8025-b92d87e0a0fc" /> | <img width="2048" height="2048" alt="Image" src="https://github.com/user-attachments/assets/cfc7e1c9-5790-4414-b1e0-f7a2bde3d3f2" /> |
| <img width="640" height="640" alt="8.webp" src="../examples/editing/data/images/8.webp" /> | <img width="1696" height="2528" alt="Image" src="https://github.com/user-attachments/assets/36ce560e-1dd1-4b10-8aee-2dda74003e42" /> | <img width="1696" height="2528" alt="Image" src="https://github.com/user-attachments/assets/3a1997b1-96cf-4bbd-a637-ae9793985811" /> |


## Existing Issues

A issue have been identified in the SenseNova-U1-8B-MoT-LoRA-8step-V1.0 (8 NFE), and we are actively working to resolve them. 

- Grid artifacts may occur in certain instances.
<img width="1536" height="2720" alt="Image" src="https://github.com/user-attachments/assets/7b97a6a7-4e1a-4af8-884f-5df134cfdc3b" /> 
