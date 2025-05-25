# cleanup_test_data.py

import os
import shutil

# 삭제 대상 JSON 파일
json_files = [
    "registered_vectors.json",
    "registered_speaker.json",
    "registered_keyword_vectors.json",
    "proto_label_map.json"
]


# 사용자 커스텀 키워드 오디오 디렉토리
custom_data_dir = "data/custom"

def delete_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"[삭제 완료] {filepath}")
    else:
        print(f"[건너뜸] {filepath} 없음")

def delete_folder_contents(folder_path):
    if not os.path.exists(folder_path):
        print(f"[건너뜸] {folder_path} 없음")
        return
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"[삭제 완료] {file_path}")

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"[폴더 삭제 완료] {folder_path}")
    else:
        print(f"[건너뜸] {folder_path} 없음")

# 실행
if __name__ == "__main__":
    print("📁 JSON 파일 삭제:")
    for jf in json_files:
        delete_file(jf)

    print("\n📁 테스트 오디오 파일 삭제:")
    delete_folder_contents(test_audio_dir)

    print("\n📁 사용자 키워드 오디오 전체 삭제:")
    delete_folder(custom_data_dir)
