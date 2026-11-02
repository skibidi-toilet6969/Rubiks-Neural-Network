# overnight
python rubiks_ai.py train --iters 200000 --batch 4096 --hidden 512 --blocks 4 --max_depth 26 --save cube_model.pt --device cuda
# 3x3x3
python rubiks_ai.py train --iters 50000 --batch 2048 --hidden 512 --blocks 3 --max_depth 26 --save cube_model.pt --device cuda
# 4x4x4
python rubiks_ai_custom_size.py train --size 4 --iters 20000 --batch 2048 --max_depth 26 --hidden 512 --blocks 4 --save cube_model_4x4x4.pt --device cuda
# 2x2x2
python rubiks_ai_custom_size.py train --size 2 --iters 20000 --batch 2048 --max_depth 11 --hidden 512 --blocks 4 --save cube_model_2x2x2.pt --device cuda

# Example For Solving Command
python rubiks_ai.py solve --model cube_model.pt --scramble "U F D2 B2 U2 F D2 F L2 B2 L2 F' L' D2 F L' D' R' F U'" --beam 4096 --device cuda
