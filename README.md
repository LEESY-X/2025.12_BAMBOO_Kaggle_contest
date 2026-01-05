# **Amazon 제품 가격 예측 경진대회**

### **Project Summary**

- **대회 개요**
    
    **웹사이트**: https://www.kaggle.com/competitions/amazonbamboo/data
    
    **목표:** 제품 정보(제품명, 카테고리, 설명, 사양, 이미지)를 기반으로 Amazon 제품의 판매 가격을 정확하게 예측하는 모델 개발
    
    **평가:**  Mean Absolute Error (MAE), 예측 가격과 실제 가격 간의 절대 오차의 평균, Public 점수는 전체 테스트 데이터의 약 30%로 계산, Private 점수는 나머지 70%로 계산
    
    **사용 데이터**: amazonproduct_train.csv - 학습 데이터(22,873개), amazonproduct_test.csv - 테스트 데이터(6,335개), sample_submission.csv - 제출 파일 예시, baseline.ipynb - 베이스라인 파일
    
- **우리 팀(DnA 1) 코드 설명**
    - **Colab의 GPU 환경에서 처리**
        - **데이터 로드 + 기본 전처리**
            
            ```python
            import pandas as pd
            import numpy as np
            import re
            import joblib
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import LabelEncoder
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
            import warnings
            
            warnings.filterwarnings('ignore')
            
            # ==========================================
            # 1. 데이터 로드 + 기본 전처리
            # ==========================================
            df_train = pd.read_csv('amazonproduct_train.csv')
            df_test = pd.read_csv('amazonproduct_test.csv')
            submission = pd.read_csv('sample_submission.csv')
            
            # 가격 전처리
            df_train['Selling Price'] = df_train['Selling Price'].astype(str).str.replace(r'[$,]', '', regex=True)
            df_train['Selling Price'] = pd.to_numeric(df_train['Selling Price'], errors='coerce')
            df_train = df_train.dropna(subset=['Selling Price'])
            y_tr = df_train['Selling Price'].values 
            
            # 결측치 채우기
            text_cols = ['Category', 'Product Specification', 'Product Name', 'Description']
            for col in text_cols:
                df_train[col] = df_train[col].fillna('Unknown').astype(str)
                df_test[col] = df_test[col].fillna('Unknown').astype(str)
            ```
            
        - **피쳐 엔지니어링**
            1. **Section [A]: 단위 통일 및 숫자 추출** 
                
                주어진 데이터 셋에서 Price와 상관관계 분석을 통해 중요한 특성들을 추출함. 
                
                - **extract_weight 함수**: Product Specification의 단위를 파운드(lb) 기준으로 통일.
                - **Item Weight**: 순수 제품 무게를 찾음.
                - **Shipping Weight**: 배송 무게를 찾음.
                - **Product Volume**: 가로*세로*높이를 찾아서 계산함
                - **Age**: 월/년 단위를 통합하여 사용연령을 계산함.
                - **Pack Qty** : Product name에서 단품/세트 상품을 확인함.
            2. **Section [B]: BERT용 요약 텍스트 생성**
                
                텍스트 분석 모델(BERT 등)에 넣을 정리된 문장 제작.
                
                - **Brand:** Manufacturer 뒤에 있는 단어를 브랜드로 인식
                - **extracted_part:** 위에서 구한 무게, 부피, 나이 등의 핵심 정보를 붙임.
            
            ```python
            # ==========================================
            # 2. 피처 엔지니어링
            # ==========================================
            def process_all_features_in_one_pass(row):
                spec = str(row['Product Specification'])
                name = str(row['Product Name'])
            
                spec_lower = spec.lower()
                name_lower = name.lower()
            
                # ---------------------------------------------------------
                # [A] 수치 정보 추출 (무게, 배송무게, 크기, 연령, 개수)
                # ---------------------------------------------------------
            
                # 함수: 텍스트에서 숫자+단위 찾아서 lb(파운드)로 변환
                def extract_weight(text, pattern_prefix=''):
                    # pattern_prefix: "shipping weight" 같은 특정 문맥 뒤를 찾을 때 사용
                    w_lb = 0.0
                    # 정규식: (접두사)...(숫자)...(단위)
                    # 예: shipping weight: 5.2 pounds
                    regex_base = pattern_prefix + r':?\s*(\d+\.?\d*)\s*'
            
                    m_lb = re.search(regex_base + r'(pound|lb)', text)
                    m_oz = re.search(regex_base + r'(ounce|oz)', text)
                    m_kg = re.search(regex_base + r'(kg|kilogram)', text)
            
                    if m_lb: w_lb = float(m_lb.group(1))
                    elif m_oz: w_lb = float(m_oz.group(1)) / 16.0
                    elif m_kg: w_lb = float(m_kg.group(1)) * 2.20462
                    return w_lb
            
                # 1. Item Weight (제품 무게)
                item_weight_lb = extract_weight(spec_lower, r'item\s*weight')
                if item_weight_lb == 0: # item weight 명시가 없으면 일반 패턴 검색
                    item_weight_lb = extract_weight(spec_lower, r'')
            
                # 2. Shipping Weight (배송 무게)
                ship_weight_lb = extract_weight(spec_lower, r'shipping\s*weight')
            
                # 3. Product Volume (크기 -> 부피 계산) 
                # 패턴: 10 x 5 x 2 (inches 생략 가능)
                vol = 0.0
                # "숫자 x 숫자 x 숫자" 패턴 찾기
                dim_m = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', spec_lower)
                if dim_m:
                    try:
                        # 가로 * 세로 * 높이
                        vol = float(dim_m.group(1)) * float(dim_m.group(2)) * float(dim_m.group(3))
                    except:
                        vol = 0.0
            
                # 4. Age (연령)
                min_age = -1.0
                age_str = None
                age_m = re.search(r'(\d+)\s*(year|yr|month)', spec_lower)
                if age_m:
                    val = float(age_m.group(1))
                    unit = age_m.group(2)
                    if 'month' in unit: min_age = val / 12.0
                    else: min_age = val
                    age_str = f"{min_age:.1f} years"
            
                # 5. Pack (수량)
                pack_qty = 1.0
                pack_m = re.search(r'(\d+)\s*(pack|pk|pcs|set|count)', name_lower)
                if pack_m: pack_qty = float(pack_m.group(1))
            
                # ---------------------------------------------------------
                # [B] BERT용 텍스트 정보 구성 
                # ---------------------------------------------------------
                extracted_parts = []
                brand_col = "Unknown"
            
                # Brand
                brand_match = re.search(r'(Manufacturer|Brand):?\s*([^|]+)', spec, re.I)
                if brand_match:
                    found_brand = brand_match.group(2).strip()
                    extracted_parts.append(f"Brand: {found_brand}")
                    brand_col = found_brand
                else:
                    if len(name) > 0: brand_col = name.split()[0]
            
                # 스펙 텍스트 조립
                if item_weight_lb > 0: extracted_parts.append(f"Weight: {item_weight_lb:.2f} lb")
                if ship_weight_lb > 0: extracted_parts.append(f"Ship Weight: {ship_weight_lb:.2f} lb") # 텍스트에도 추가
                if vol > 0: extracted_parts.append(f"Vol: {vol:.2f}") 
                if age_str: extracted_parts.append(f"Age: {age_str}")
            
                cleaned_spec = " | ".join(extracted_parts) if extracted_parts else "Unknown"
            
                # 반환값에 새로 만든 수치형 변수들(item_weight, ship_weight, vol) 추가
                return pd.Series([cleaned_spec, brand_col, min_age, item_weight_lb, ship_weight_lb, vol, pack_qty],
                                 index=['Cleaned_Spec', 'Brand', 'min_age', 'item_weight_lb', 'ship_weight_lb', 'product_vol', 'pack_qty'])
            
            # 적용
            new_cols = ['Cleaned_Spec', 'Brand', 'min_age', 'item_weight_lb', 'ship_weight_lb', 'product_vol', 'pack_qty']
            df_train[new_cols] = df_train.apply(process_all_features_in_one_pass, axis=1)
            df_test[new_cols] = df_test.apply(process_all_features_in_one_pass, axis=1)
            ```
            
        - **인코딩 및 수치형 변수 정리**
            
            인코딩: 글자로 구성된 카테고리 정보(범주형) → 숫자로 변환(모델에게 제공)
            
            1. 카테고리 세분화: 대분류|중분류|소분류|…의 문자열을 3단계의 독립된 컬럼으로 쪼갬.
            2. 레이블 인코딩: 정보를 구별하기 위해 카테고리에 번호 부여.
            3. 타겟 인코딩: 브랜드 평균을 계산후 가격 평균 값으로 해당 브랜드 치환.
            4. 최종 수치형 데이터 셋: 물리적 속성(피처엔지니어링), 브랜드 가격 평균(타겟 인코딩), 고유 ID(카테고리 세분화+레이블 인코딩)을 하나로 묶음.
                - 모델은 카테고리 그룹과 가격 수준의 연결성을 식별함.
            
            ```python
            # ==========================================
            # 3. 인코딩 및 수치형 변수 정리
            # ==========================================
            
            # 1) 카테고리 세분화
            def split_category(df):
                split_data = df['Category'].str.split('|', n=2, expand=True)
                df['Main_Cat'] = split_data[0].fillna('Unknown').str.strip()
                df['Sub_Cat'] = split_data[1].fillna('Unknown').str.strip()
                df['Deep_Cat'] = split_data[2].fillna('Unknown').str.strip() if split_data.shape[1] > 2 else 'Unknown'
                return df
            
            df_train = split_category(df_train)
            df_test = split_category(df_test)
            
            # 2) 레이블 인코딩
            label_cols = ['Brand', 'Main_Cat', 'Sub_Cat', 'Deep_Cat']
            for col in label_cols:
                le = LabelEncoder()
                all_values = pd.concat([df_train[col], df_test[col]]).astype(str).unique()
                le.fit(all_values)
                df_train[f'{col}_label'] = le.transform(df_train[col].astype(str))
                df_test[f'{col}_label'] = le.transform(df_test[col].astype(str))
            
            # 3) 타겟 인코딩 
            target_cols = ['Brand', 'Main_Cat', 'Sub_Cat', 'Deep_Cat'] 
            global_mean = y_tr.mean()
            
            for col in target_cols:
                mean_map = df_train.groupby(col)['Selling Price'].mean()
                df_train[f'{col}_target_enc'] = df_train[col].map(mean_map)
                df_test[f'{col}_target_enc'] = df_test[col].map(mean_map).fillna(global_mean)
            
            # 최종 수치형 데이터셋
            final_num_cols = ['min_age', 'pack_qty', 'item_weight_lb', 'ship_weight_lb', 'product_vol'] + \
                             [f'{c}_target_enc' for c in target_cols] + \
                             [f'{c}_label' for c in label_cols]
            
            print(f"   -> Numerical Features ({len(final_num_cols)}): {final_num_cols}")
            
            X_num_tr = df_train[final_num_cols].values
            X_num_te = df_test[final_num_cols].values
            ```
            
        - **SOTA 임베딩 + PCA 차원 축소+ Clustering 군집화**
            1. 임베딩: SentenceTransformer (BERT)를 활용해 텍스트 → 문장 하나당 숫자 벡터(768개)로 변환.
            2. 차원 축소: 768개의 숫자를 64개의 차원으로 축소.
            3. 군집화: 비슷한 의미를 가진 문장끼리 묶음 → 그룹 번호(범주형 변수)를 매김
            4. 적용
                - Product Name: r_n (임베딩+차원축소+군집화)
                - Description: tr_d (임베딩+차원축소+군집화)
                - Category + Product Specification: tr_c (최종 수치형 데이터 셋에 카테고리 추가후 임베딩+차원축소+군집화)
                    - 모델은 카테고리의 문맥과 뉘양스를 뽑아냄.
            
            ```python
            # ==========================================
            # 4. SOTA 임베딩 + PCA + 클러스터링
            # ==========================================
            bert = SentenceTransformer('all-mpnet-base-v2')
            
            def get_advanced_emb(text_list_tr, text_list_te, n_comp, n_cluster):
                emb_tr = bert.encode(text_list_tr, show_progress_bar=True, batch_size=64)
                emb_te = bert.encode(text_list_te, show_progress_bar=True, batch_size=64)
            
                pca = PCA(n_components=n_comp, random_state=42)
                pca_tr = pca.fit_transform(emb_tr)
                pca_te = pca.transform(emb_te)
            
                kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=10)
                cluster_tr = kmeans.fit_predict(pca_tr).reshape(-1, 1)
                cluster_te = kmeans.predict(pca_te).reshape(-1, 1)
            
                return np.hstack([pca_tr, cluster_tr]), np.hstack([pca_te, cluster_te])
            
            # 1. Product Name 
            print("   -> Processing Product Name...")
            tr_n, te_n = get_advanced_emb(df_train['Product Name'].tolist(),
                                          df_test['Product Name'].tolist(), 64, 20)
            
            # 2. Description
            print("   -> Processing Description...")
            tr_d, te_d = get_advanced_emb(df_train['Description'].tolist(),
                                          df_test['Description'].tolist(), 64, 20)
            
            # 3. Category + Cleaned_Spec
            print("   -> Processing Category + Spec...")
            cats_tr = (df_train['Category'] + " " + df_train['Cleaned_Spec']).tolist()
            cats_te = (df_test['Category'] + " " + df_test['Cleaned_Spec']).tolist()
            tr_c, te_c = get_advanced_emb(cats_tr, cats_te, 64, 20)
            ```
            
        - **저장**
            
            최종 수치형 변수, 임베딩+차원축소+군집화 결과들 → 최종 피처 개수 출력 및 저장
            
            ```python
            # ==========================================
            # 5. 저장
            # ==========================================
            X_tr = np.hstack([tr_n, tr_d, tr_c, X_num_tr])
            X_test = np.hstack([te_n, te_d, te_c, X_num_te])
            
            print(f"최종 피처 개수: {X_tr.shape[1]}")
            
            package = {
                'X_tr': X_tr,
                'y_tr': y_tr,
                'X_test': X_test,
                'submission': submission,
                'feature_names': num_cols + ['emb_pca_...']
            }
            save_path = "data6.pkl"
            joblib.dump(package, save_path, compress=3)
            print(f"데이터셋 저장: {save_path}")
            ```
            
    - **로컬 CPU 환경에서 처리**
        - **데이터 로드**
            
            앞서 GPU에서 처리한 데이터를 불러온 후 헬퍼 함수로 교차 검증을 위한 학습/검증 데이터 분리
            
            ```python
            import joblib
            import numpy as np
            import pandas as pd
            import lightgbm as lgb
            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import KFold
            
            # 1. 데이터 로드 
            file_path = 'data6.pkl'
            
            try:
                data = joblib.load(file_path)
                X_tr = data['X_tr']
                y_tr = data['y_tr']
                X_test = data['X_test']
                submission = data['submission']
                print(f"성공! 피처 개수: {X_tr.shape[1]}개")
            except FileNotFoundError:
                print("에러: 파일을 찾을 수 없습니다.")
                exit()
            
            # 헬퍼 함수
            def get_fold_data(X, y, train_idx, val_idx):
                if hasattr(X, 'iloc'): X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
                else: X_t, X_v = X[train_idx], X[val_idx]
                if hasattr(y, 'iloc'): y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
                else: y_t, y_v = y[train_idx], y[val_idx]
                return X_t, X_v, y_t, y_v
            ```
            
        - **LightGBM 학습+후처리**
            - **eval_rounded_mae_lgb 함수**: 모델은 로그된 값을 예측하지만, 실제 채점 기준은 “원래 가격을 소수점 둘째 자리에서 반올림한 값의 오차(MAE)”로 사용하기에 따로 정의후 사용.
            - **np.log1p(y)** : 가격데이터는 보통 싼 물건이 많고 비싼 물건은 적은 분포이기에 log를 씌워 정규분포처럼 펴준 뒤 학습하고, 나중에 expm1으로 되돌림.
            - **K-Fold 교차 검증**: 데이터를 5조각(n_splits=5)으로 나누어 5번 학습.
            - **하이퍼파라미터 설정**
                - objective: 'regression' (
                - metric: 'None' (우리가 만든 커스텀 함수 사용)
                - learning_rate: 0.008
                - num_boost_round: 100000 + Early Stopping 2000
            - **후처리**: 반올림을 하는게 더 좋은지 확인후 최종 결과 산출.
            - **안전장치**: 가격은 음수가 될 수 없으므로, 최소값을 0.01로 강제 고정하여 에러를 방지.
            
            ```python
            import lightgbm as lgb
            from sklearn.model_selection import KFold
            from sklearn.metrics import mean_absolute_error
            import numpy as np
            import pandas as pd
            
            # ====================================================
            # 1. 사용자 정의 평가 함수 (Rounded MAE)
            # ====================================================
            def eval_rounded_mae_lgb(preds, train_data):
                labels = train_data.get_label()
                preds = np.round(np.expm1(preds), 2)
                actual = np.expm1(labels)
                return 'rounded_mae', mean_absolute_error(actual, preds), False
                
            # ====================================================
            # 2. LightGBM 학습 및 평가 
            # ====================================================
            def run_lgbm_final(X, y, X_test, submission_df, n_splits=5):
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                
                # 결과 저장용 배열
                oof_preds = np.zeros(X.shape[0])
                test_preds = np.zeros(X_test.shape[0])
                scores = []
            
                params = {
                    'objective': 'regression',
                    'metric': 'None',
                    'boosting_type': 'gbdt',
                    'num_leaves': 64,
                    'learning_rate': 0.008,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 3,
                    'verbose': -1,
                    'n_jobs': -1,
                    'random_state': 42
                }
                
                # Log 변환 (타겟)
                y_log = np.log1p(y)
                
                # 피처 중요도 출력을 위한 이름 리스트
                feat_names = [f'Feature {i}' for i in range(X.shape[1])]
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                    # 데이터 분할 
                    try:
                        X_t, X_v, y_t, y_v = get_fold_data(X, y_log, train_idx, val_idx)
                    except NameError:
                        X_t, X_v = X[train_idx], X[val_idx]
                        y_t, y_v = y_log[train_idx], y_log[val_idx]
                    
                    train_ds = lgb.Dataset(X_t, label=y_t)
                    valid_ds = lgb.Dataset(X_v, label=y_v, reference=train_ds)
                    
                    # 모델 학습
                    model = lgb.train(
                        params, 
                        train_ds, 
                        num_boost_round=100000,
                        valid_sets=[valid_ds],
                        feval=eval_rounded_mae_lgb,  # 커스텀 평가 함수 적용
                        callbacks=[
                            lgb.early_stopping(2000, verbose=False),
                            lgb.log_evaluation(1000) 
                        ]
                    )
                    
                    # 예측 및 역변환
                    val_pred_log = model.predict(X_v)
                    test_pred_log = model.predict(X_test)
                    
                    val_pred = np.expm1(val_pred_log)
                    test_pred = np.expm1(test_pred_log)
                    
                    oof_preds[val_idx] = val_pred
                    test_preds += test_pred / n_splits
                    
                    # 점수 계산 (반올림 적용된 점수로 기록)
                    best_score = model.best_score['valid_0']['rounded_mae']
                    scores.append(best_score)
                    
                    print(f"    -> Fold {fold+1} Best Rounded MAE: {best_score:.4f}")
                    
                # ==========================================
                # 6. 최종 결과 평가 및 저장 로직
                # ==========================================
                # 1) 원본 OOF 점수
                score_final = mean_absolute_error(y, oof_preds)
                
                # 2) 반올림 적용 OOF 점수
                score_rounded = mean_absolute_error(y, np.round(oof_preds, 2))
                
                print("\n" + "="*50)
                print(f" LightGBM Global MAE (Raw): {score_final:.5f}")
                print(f" 반올림 적용 시 MAE : {score_rounded:.5f}")
                
                use_rounding = False
                if score_rounded < score_final:
                    print("반올림(소수점 2자리)이 더 유리하여 적용했음.")
                    use_rounding = True
                else:
                    print("반올림하지 않는 것이 더 좋음.")
                print("="*50)
                
                # 최종 예측값 후처리
                final_preds = np.maximum(0.01, test_preds) # 음수 방지
                
                if use_rounding:
                    final_preds = np.round(final_preds, 2)
                    final_score_str = f"{score_rounded:.4f}"
                else:
                    final_score_str = f"{score_final:.4f}"
                
                # 파일 저장
                save_filename = f'LGBM_Best_MAE_{final_score_str}.csv'
                submission_df['Selling Price'] = final_preds
                submission_df.to_csv(save_filename, index=False)
                print(f"\n 파일 저장 완료: {save_filename}")
                
                return final_preds
            final_results = run_lgbm_final(X_tr, y_tr, X_test, submission)
            
            ```
            
    
- **최종 결과 및 Insight**
    - **최종 결과**
        
        Public score: 1.75449
        
        Private score: 1.98977
        
    - **다른 조의 아이디어**
        
        Rule-based Model 이용
        
        파이프라인형 앙상블: Lookup/Rule-based+Ridge Regression+kNN
        
        TF-IDF 벡터화를 이용한 임베딩
        
    - **Insight**
        
        모델 성능은 데이터 이해에서 시작된다.
        
        모든 문제를 복잡한 모델로 풀 필요는 없다.
        
        사람이 가격을 판단한다면 어떻게 할까라는 관점으로 접근했다.
        
        평균의 함정을 벗어나기 위해서 데이터 가격 분포로 나눠서 매우 작은 값과 큰 값들을 분리했다. 
        
        조원들이 각자 모델링, 데이터 처리, 파라미터 튜닝을 각각 분배하여 진행했다.
        
        쓸모 없어 보이는 데이터도 어떻게든 사용하기 위해 노력했다.
