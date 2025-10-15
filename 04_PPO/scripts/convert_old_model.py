"""
기존 모델을 새 네트워크 구조로 변환
"""
import torch
import sys

def convert_model(old_path, new_path):
    print(f"변환 중: {old_path} -> {new_path}")
    
    # 기존 체크포인트 로드
    checkpoint = torch.load(old_path, map_location='cpu')
    
    # 키 이름 매핑
    old_to_new = {
        'x_direction_head.weight': 'actor_x.weight',
        'x_direction_head.bias': 'actor_x.bias',
        'y_direction_head.weight': 'actor_y.weight',
        'y_direction_head.bias': 'actor_y.bias',
        'power_hit_head.weight': 'actor_power.weight',
        'power_hit_head.bias': 'actor_power.bias',
    }
    
    # 새 state_dict 생성
    new_state_dict = {}
    for old_key, value in checkpoint['network_state_dict'].items():
        new_key = old_to_new.get(old_key, old_key)
        new_state_dict[new_key] = value
        if old_key in old_to_new:
            print(f"  {old_key} -> {new_key}")
    
    # 새 체크포인트 생성
    new_checkpoint = {
        'network_state_dict': new_state_dict,
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        'total_timesteps': checkpoint['total_timesteps'],
        'num_updates': checkpoint['num_updates'],
    }
    
    # 저장
    torch.save(new_checkpoint, new_path)
    print(f"\n변환 완료! 저장: {new_path}")
    print(f"Total timesteps: {checkpoint['total_timesteps']:,}")
    print(f"Num updates: {checkpoint['num_updates']:,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_old_model.py <old_model.pth> [new_model.pth]")
        sys.exit(1)
    
    old_path = sys.argv[1]
    new_path = sys.argv[2] if len(sys.argv) > 2 else old_path.replace('.pth', '_converted.pth')
    
    convert_model(old_path, new_path)

