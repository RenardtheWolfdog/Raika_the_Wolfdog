# Raika_S3.py

import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from urllib.parse import quote
import configparser
import asyncio

class S3Handler:
    def __init__(self, bucket_name, region_name='ap-northeast-2'):
        config = configparser.ConfigParser()
        try:
            config.read('config.ini', encoding='utf-8')
        except Exception:
            config.read('config.ini')
        
        session = boto3.Session(
            aws_access_key_id=config['AWS']['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=config['AWS']['AWS_SECRET_ACCESS_KEY'],
            region_name=region_name
        )
        self.s3 = session.client('s3')
        self.bucket_name = bucket_name
        self.region_name = region_name

    def upload_file(self, file_path, object_name):
        try:
            # 파일이 존재하는지, 비어있지 않은지 확인
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' does not exist.")
                return False
            if os.path.getsize(file_path) == 0:
                print(f"Error: File '{file_path}' is empty.")
                return False

            self.s3.upload_file(file_path, self.bucket_name, object_name)
            print(f"Successfully uploaded '{file_path}' to '{object_name}'")
            return True
        except NoCredentialsError:
            print("자격 증명을 찾을 수 없습니다.")
            return False
        except Exception as e:
            print(f"업로드 중 오류 발생: {str(e)}")
            return False
        
    def get_file_url(self, object_name):
        try:
            # URL에 안전한 형식으로 인코딩
            encoded_object_name = quote(object_name)
            url = f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{encoded_object_name}"

            return url
        except Exception as e:
            print(f"Error generating URL for '{object_name}': {str(e)}")
            return None

    def delete_objects(self, object_keys):
        try:
            # 빈 키나 None 값 제거
            valid_keys = [key for key in object_keys if key]
            if not valid_keys:
                return True # 삭제할 키가 없으면 성공으로 간주함

            encoded_keys = [{'Key': quote(key, safe='')} for key in valid_keys]
            response = self.s3.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': encoded_keys}
            )
            if 'Errors' in response:
                print(f"일부 객체 삭제 실패: {response['Errors']}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'MalformedXML':
                print("XML 형식 오류. 객체 키를 확인하세요.")
            else:
                print(f"S3 클라이언트 오류: {e}")
            return False
        except Exception as e:
            print(f"객체 삭제 중 예상치 못한 오류 발생: {str(e)}")
            return False
        
    # def delete_folder(self, prefix):
    #     try:
    #         # 해당 접두사(세션 ID)로 시작하는 모든 객체 나열
    #         objects_to_delete = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

    #         # 객체가 존재하면 삭제
    #         if 'Contents' in objects_to_delete:
    #             delete_keys = {'Objects': [{'Key': obj['Key']} for obj in objects_to_delete['Contents']]}
    #             self.s3.delete_objects(Bucket=self.bucket_name, Delete=delete_keys)
    #             print(f"Deleted S3 folder: {prefix}")
    #             return True
    #         else:
    #             print(f"No objects found in S3 folder: {prefix}")
    #             return True
    #     except ClientError as e:
    #         print(f"Error deleting S3 folder {prefix}: {e}")
    #         return False

    def read_object(self, object_name):
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_name)
            content = response['Body'].read()
            return content
        except Exception as e:
            print(f"객체 읽기 중 오류 발생: {str(e)}")
            return None

    """연동 확인을 위해 버킷 속 객체를 읽어들임"""
    def list_objects(self, prefix=''):
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            print(f"객체 목록 조회 중 오류 발생: {str(e)}")
            return []
        

# FastAPI를 위한 추가 비동기 메서드들

import logging

class AsyncS3Handler(S3Handler):
    """S3Handler의 비동기 버전"""

    async def async_upload_file(self, file_path, object_name):
        """파일 업로드 (비동기 버전)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.upload_file, file_path, object_name)
    
    async def async_get_file_url(self, object_name):
        """파일 URL 생성 (비동기 버전)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_file_url, object_name)
    
    async def async_delete_objects(self, object_keys):
        """객체 삭제 (비동기 버전)"""
        if not object_keys:
            return True # 빈 리스트면 성공으로 간주

        loop = asyncio.get_event_loop()
        try:
            # 객체 키를 직접 사용하는 대신 딕셔너리 리스트로 변환
            delete_objects = [{'Key': key} for key in object_keys]

            # 삭제 요청 함수 정의
            def delete_action():
                try:
                    response = self.s3.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': delete_objects}
                    )
                    if 'Errors' in response and response['Errors']:
                        logging.error(f"S3 deletion errors: {response['Errors']}")
                        return False
                    return True
                except Exception as e:
                    import traceback
                    logging.error(f"Error in delete_objects: {str(e)}\n{traceback.format_exc()}")
                    return False
            
            # 실행
            return await loop.run_in_executor(None, delete_action)

        except Exception as e:
            import traceback
            logging.error(f"Error in async_delete_objects: {str(e)}\n{traceback.format_exc()}")
            return False

    async def async_read_object(self, object_name):
        """객체 읽기 (비동기 버전)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_object, object_name)
    
    async def async_list_objects(self, prefix=''):
        """객체 목록 조회 (비동기 버전)"""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self.list_objects, prefix)
            return result # 리스트 반환
        except Exception as e:
            import traceback
            logging.error(f"Error in async_list_objects: {str(e)}\n{traceback.format_exc()}")
            return [] # 오류 발생 시 빈 리스트 반환
                
if __name__ == "__main__":
    # S3Handler 인스턴스 생성
    s3_handler = S3Handler('imageandvediobucket')

    # 버킷 내 객체 목록 조회
    objects = s3_handler.list_objects()
    print("버킷 내 객체 목록:")
    for obj in objects:
        print(f" - {obj}")