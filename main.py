import image_change
import model

if __name__=="__main__":
    
    # 이미지 변환 수행 
    image_change.convert_start() 
    
    # 이미지 전처리 및 모델 생성, 학습 진행 
    model.model_ImageData_Gen_training()
 
