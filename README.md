# Đồ án IS403 – Tái hiện và mở rộng bài báo e-Profits

> **Bài báo gốc:** *e-profits: a business-aligned evaluation metric for profit-sensitive customer churn prediction*<br>
```bibtex
@article{manzoor2026profits,
  title={e-profits: a business-aligned evaluation metric for profit-sensitive customer churn prediction},
  author={Manzoor, Awais and Qureshi, M Atif and Kidney, Etain and Longo, Luca},
  journal={International Journal of Data Science and Analytics},
  volume={22},
  number={1},
  pages={75},
  year={2026},
  publisher={Springer}
}
```
> **Source Code:** https://github.com/Awaismanzoor/eprofits <br>
> **Dữ liệu:** IBM Telco (7.043 dòng, 19 đặc trưng, churn 26,5%) · Maven Telecom (7.043 dòng, 32 đặc trưng, 27% churn / 67% stay / 6% new joiner)

## 0. Phạm vi và thứ tự ưu tiên

| Ưu tiên | Phần | Trạng thái |
|---|---|---|
| 1 | **Giai đoạn 1** – Tái hiện bài báo gốc | ☐ |
| 2 | **Giai đoạn 2 – Module 1** – FT-Transformer, NODE | ☐ |
| 3 | **Giai đoạn 2 – Module 2** – SMOTE + XGBoost | ☐ |
| 4 | **Giai đoạn 3 – Module 3** – Phân tích độ nhạy kinh doanh | ☐ |
| 5 | **Giai đoạn 3 – Module 4** – Ablation và nhiễu | ☐ |


---

## 1. Chuẩn bị chung 

| STT | Công việc | Chi tiết | Output |
|---|---|---|---|
| 1 | Đọc kỹ bài báo | Mục 3 (công thức Eq.1–8), Bảng 2 (lưới siêu tham số), Bảng 3–4 (kết quả), Phụ lục A | Ghi chú tóm tắt công thức + tham số |
| 2 | Đọc code gốc | Clone repo, đọc 4 file `.py` / notebook của IBM và Maven. **Ghi lại những gì bài báo không nói rõ**: tỷ lệ chia train/test, `random_state`, `stratify`, giá trị `c0`, `c1` trong thí nghiệm chính (phụ lục chỉ dùng `c0=5, c1=0.3` cho ví dụ), cách chọn "top 20%", đơn vị tenure | Các "tham số bài báo không nêu" (xem mục 7) |
| 3 | Dựng môi trường | Python 3.10+, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `interpret` (EBM), `lifelines` (Kaplan–Meier), `imbalanced-learn`, `torch`, `pytorch_tabular` hoặc `rtdl`, `optuna`. Khoá phiên bản bằng `requirements.txt`. Chạy trên Colab/Kaggle GPU cho phần deep learning | `requirements.txt`, môi trường chạy được |
| 4 | Cấu trúc repo nhóm | Xem sơ đồ bên dưới | Repo GitHub của nhóm |
| 5 | Cố định seed & log | Seed toàn cục, lưu mọi kết quả ra `results/*.csv` để dựng bảng báo cáo mà không phải chạy lại | `utils/seed.py`, thư mục `results/` |
| 6 | Phân công | Gợi ý: (A) tiền xử lý + KM + e-Profits, (B) 6 mô hình gốc + tuning, (C) FT-T/NODE, (D) SMOTE + so sánh, (E) báo cáo + slide | Bảng phân công |

```
project/
├── data/            # IBM.csv, Maven.csv (bản gốc)
├── src/
│   ├── preprocess.py      # tiền xử lý, chia train/test
│   ├── survival.py        # Kaplan–Meier, TRR/ARR
│   ├── eprofits.py        # CLV, chi phí, e-Profits (có unit test)
│   ├── models_base.py     # RF, GBC, XGB, LGBM, CBC, EBM + lưới siêu tham số
│   ├── models_ext.py      # FT-Transformer, NODE
│   ├── smote_exp.py       # Module 2
│   └── evaluate.py        # Acc, F1, AUC, Lift, e-Profits, bootstrap
├── notebooks/
├── results/
├── tests/test_eprofits_toy.py
└── README.md
```

---

## 2. GIAI ĐOẠN 1 – Tái hiện bài báo gốc

**Mục tiêu:** Chạy lại đúng quy trình và đạt kết quả tương đương Bảng 3, 4, 5 của bài báo (chấp nhận sai số do seed/split).

| STT | Công việc | Chi tiết / điểm cần cẩn thận | Output | Trạng thái |
|---|---|---|---|---|
| 1 | Tải & kiểm tra dữ liệu | IBM: kiểm tra `TotalCharges` (bản Kaggle thường có một số ô rỗng dù bài báo ghi "không có missing"). Maven: kiểm tra các giá trị `Customer Status` | Báo cáo EDA ngắn: kích thước, tỷ lệ churn, kiểu dữ liệu | ☐ |
| 2 | Tiền xử lý IBM | Chuyển nhãn `Churn` → 0/1; mã hoá biến phân loại; giữ lại cột **doanh thu hàng tháng** (`MonthlyCharges`) và **tenure** để tính CLV/KM (đây là `R_i` và `T_i`) | `IBM_clean.csv` | ☐ |
| 3 | Tiền xử lý Maven | **Loại 6% "new joiners"** (Customer Status = Joined) để về bài toán nhị phân. **Loại các cột rò rỉ nhãn** như `Churn Category`, `Churn Reason` (chỉ có giá trị khi đã churn) – đối chiếu với code gốc xem tác giả xử lý thế nào. Với các cột định danh/địa lý (ID, Zip, Lat/Long, City) quyết định giữ hay bỏ và ghi lý do | `Maven_clean.csv`, bảng cột đã loại | ☐ |
| 4 | Chia train/test **một lần** | Chia một lần, stratify theo nhãn, cố định seed; lưu chỉ số để mọi mô hình dùng chung. Tỷ lệ theo code gốc (nếu không rõ: 80/20). **Không SMOTE, không `class_weight`** để giữ phân bố gốc | `split_idx_{ibm,maven}.npy` | ☐ |
| 5 | Kaplan–Meier (chỉ fit trên **train**) | `KaplanMeierFitter(duration=tenure, event=churn)`. Churn = sự kiện, chưa churn = censored. Tính `S(t)`. **Chống leakage:** không fit trên toàn bộ dữ liệu; KM **fit một lần**, giữ cố định trong tuning (không fit lại theo fold) | Hàm `survival.py`, đường cong KM (đối chiếu Hình 2a, 3a) | ☐ |
| 6 | Tính TRR & ARR | `r_TRR_i = S(T_i + 1) / S(T_i)` (Δ = 1 tháng); `r_ARR = mean(r_TRR)`; **cap `min(r, 0.995)`**. Mốc kiểm tra: **ARR ≈ 0.77 (IBM), ≈ 0.75 (Maven)** | Cột `r_TRR`, hằng số `r_ARR` | ☐ |
| 7 | Cài đặt e-Profits | `CLV = R·M / (1 − min(r, 0.995))` với `M = 0.3`; `C_offer = CPO·CLV` (`CPO = 0.1`); `C_contact = max(c0, c1·C_offer)`; lợi nhuận mỗi khách: TP = `CLV − C_offer − C_contact`, FP = `−C_offer − C_contact`, còn lại = 0; ngưỡng phân lớp **0.5**. Hỗ trợ `η = 1.0` (toàn bộ) và `η = 0.2` (top 20%) × `{ARR, TRR}` | `eprofits.py` | ☐ |
| 8 | **Unit test bằng ví dụ Phụ lục A** | 5 khách hàng A–E (Bảng 9). Kết quả phải ra đúng **228,8 (TRR)** và **308,75 (ARR)** với `M=0.3, CPO=0.1, c0=5, c1=0.3`. Đây là bước kiểm chứng công thức quan trọng nhất trước khi chạy mô hình | `tests/test_eprofits_toy.py` pass | ☐ |
| 9 | Cài đặt các thang đo | Accuracy, F1, AUC, Top-Decile Lift, Lift Index, e-Profits (4 biến thể), EMP (tuỳ chọn, ưu tiên thấp – cần thư viện/tự cài với `CLV=200, C_offer=10, C_contact=1`) | `evaluate.py` | ☐ |
| 10 | Chuẩn bị 6 mô hình + lưới siêu tham số | RF, GBC, XGB, LGBM, CBC, EBM theo **Bảng 2** của bài báo (RF 108 tổ hợp, XGB 54, GBC 27, LGBM 8, CBC 8, EBM 24) | `models_base.py` | ☐ |
| 11 | Scorer e-Profits cho `GridSearchCV` | Viết `make_scorer` tự định nghĩa: nhận chỉ số fold, tra `R_i`, `r_i` của các khách trong fold, tính e-Profits. KM giữ cố định. Cần cách truyền `R`, `r` theo index (ví dụ giữ DataFrame và dùng `X.index`) | Scorer kiểm tra được | ☐ |
| 12 | Huấn luyện + tuning | `GridSearchCV`, 5-fold CV **chỉ trên train**, với các mục tiêu tuning: AUC, Accuracy, F1, **e-Profits (TRR, toàn bộ)**; EMP làm sau nếu có thời gian. Sau tuning refit trên toàn bộ train, **đánh giá một lần trên test**. EBM chậm → dùng `n_jobs`/GPU, chạy qua đêm | `results/phase1_{ibm,maven}.csv` | ☐ |
| 13 | Dựng lại Bảng 3, 4, 5 | Cùng định dạng cột như bài báo (F1, Acc, AUC, TDL, Lift index, e-Prof ×4, EMP) | `table3_repro.csv`, `table4_repro.csv`, `table5_repro.csv` | ☐ |
| 14 | (Nên làm) Thống kê | Bootstrap 500 lần trên test → CI 95% (Bảng 6); Wilcoxon (Bảng 7); Spearman giữa xếp hạng AUC và xếp hạng profit (Bảng 8, kỳ vọng ρ ≈ 0.4–0.5) | `table6–8_repro.csv` | ☐ |
| 15 | Đối chiếu với bài báo | Bảng **Bài báo vs Tự chạy lại**: chênh lệch tuyệt đối và %. Phân tích nguyên nhân lệch (seed, split khác, phiên bản thư viện, tham số không nêu) | `repro_comparison.csv` + đoạn phân tích | ☐ |
| 16 | Kiểm chứng 2 kết luận chính | (a) Khi tuning trên e-Profits, **XGBoost** có e-Profits toàn bộ cao nhất (bài báo: IBM ≈ 10,99 triệu, Maven ≈ 1,063 triệu). (b) **Mô hình AUC cao nhất chưa chắc lời nhiều nhất**: khi tuning theo AUC, EBM có AUC cao nhất nhưng RF có e-Profits cao hơn (IBM ≈ 9,10 triệu vs 7,89 triệu; Maven ≈ 997 nghìn vs 880 nghìn). Nếu số của nhóm khác, **báo cáo trung thực** thay vì ép cho khớp | Biểu đồ/bảng minh hoạ (AUC vs e-Profits), kết luận | ☐ |

---

## 3. GIAI ĐOẠN 2 – MODULE 1: FT-Transformer, NODE

**Input:** dữ liệu đã tiền xử lý và chia train/test như Giai đoạn 1 (dùng lại đúng split, KM, hàm e-Profits).

| STT | Công việc | Chi tiết | Output | Trạng thái |
|---|---|---|---|---|
| 1 | Pipeline tiền xử lý cho DL | Biến số: `QuantileTransformer`/chuẩn hoá; biến phân loại: embedding (FT-T) hoặc one-hot/ordinal (NODE). Fit bộ biến đổi **chỉ trên train** | `preprocess_dl.py` | ☐ |
| 2 | FT-Transformer | Dùng `rtdl_revisiting_models` hoặc `pytorch_tabular`. Cấu hình **nhỏ**: `d_token` 32–64, 1–3 block, 4 head, dropout 0.1–0.3, AdamW + weight decay. Early stopping | `results/m1_ftt.csv` | ☐ |
| 3 | NODE | Dùng `pytorch_tabular` (`NodeConfig`) hoặc repo gốc NODE. Cấu hình nhỏ: 1–2 layer, 128–512 cây, depth 4–6, `entmax15`. Khởi tạo theo dữ liệu (cần batch lớn cho lần đầu) | `results/m1_node.csv` | ☐ |
| 4 | Tuning công bằng | GBDT trong bài báo được tune bằng lưới nhỏ; DL dùng Optuna **20–30 trial**, 5-fold CV trên train, mục tiêu e-Profits (TRR, toàn bộ) và AUC. Ghi rõ ngân sách để so sánh minh bạch | `optuna_*.db`, bảng không gian tìm kiếm | ☐ |
| 5 | Nhiều seed | Mỗi mô hình DL chạy **5–10 seed**, báo cáo mean ± std; mô hình cuối có thể ensemble các seed | Bảng mean ± std | ☐ |
| 6 | Hiệu chuẩn xác suất cho DL | Tách một phần validation **từ train** để early stopping + hiệu chuẩn (Platt/temperature). Báo cáo Brier/ECE. (Lý do: e-Profits phụ thuộc ngưỡng 0.5 và đầu ra xác suất) | Biểu đồ reliability | ☐ |
| 7 | Đánh giá trên test | Tính đủ bộ thang đo như Giai đoạn 1 | `m1_all_models.csv` | ☐ |
| 8 | **Bảng so sánh chính** | **XGBoost (gốc) vs FT-Transformer vs NODE**: AUC, F1, Acc, e-Profits (full/top20, ARR/TRR), thời gian train/inference, số tham số | `m1_comparison.csv` + biểu đồ | ☐ |
| 9 | Phân tích | Mô hình nào lời nhất? DL có thắng XGB trên dữ liệu nhỏ? Vì sao? (liên hệ mục 3.1 bên dưới) | Đoạn phân tích trong báo cáo | ☐ |

### 3.1 Vấn đề: FT-Transformer và NODE trên dữ liệu chỉ vài nghìn dòng

Số liệu thực tế: sau khi chia 80/20, train IBM ≈ 5,6 nghìn dòng; Maven (đã bỏ new joiners, ≈ 6,6 nghìn) train ≈ 5,3 nghìn dòng, test mỗi bên chỉ khoảng 1,3–1,4 nghìn dòng với vài trăm khách churn.

| # | Vấn đề | Biểu hiện | Giải pháp / hướng giải quyết |
|---|---|---|---|
| 1 | **Thiếu dữ liệu để hội tụ ổn định** (FT-T, NODE có hàng chục nghìn đến hàng trăm nghìn tham số, được thiết kế và đánh giá chủ yếu trên tập lớn) | Train loss giảm nhưng val loss tăng sớm (overfit); kết quả dao động mạnh giữa các seed | Dùng kiến trúc **nhỏ** (ít block/ít cây, `d_token` thấp); weight decay, dropout, early stopping theo val loss; learning-rate schedule (cosine/ReduceLROnPlateau); batch 256–512; ensemble nhiều seed |
| 2 | **GBDT thường thắng DL trên dữ liệu bảng nhỏ** | FT-T/NODE thấp hơn XGBoost về AUC | Đặt kỳ vọng ngay từ đầu: **"DL không vượt XGBoost" vẫn là kết luận hợp lệ** nếu đã tune công bằng. Trọng tâm báo cáo là *phân tích vì sao* (dữ liệu nhỏ, đặc trưng chủ yếu là phân loại, tính bất biến xoay…) chứ không phải ép DL thắng |
| 3 | **Xác suất không được hiệu chuẩn + không dùng class weight + ngưỡng cố định 0.5** | DL ít dự đoán lớp churn → ít TP, e-Profits thấp/bất thường; hoặc quá tự tin → nhiều FP | Hiệu chuẩn (Platt/temperature scaling) trên validation tách từ train; báo cáo Brier/ECE; so sánh e-Profits trước/sau hiệu chuẩn. **Không** dùng class weight/SMOTE ở Module 1 để giữ nguyên tiền đề của bài báo |
| 4 | **e-Profits trên test nhỏ có phương sai lớn** (chính CI của bài báo đã chồng lấn: Maven ≈ 0,87–1,13 triệu) | Chênh lệch giữa các mô hình nằm trong sai số | Multi-seed + bootstrap CI 95% + kiểm định cặp (Wilcoxon); không kết luận "A hơn B" khi CI chồng lấn |
| 5 | **Tiền xử lý đặc trưng ảnh hưởng mạnh tới DL** (DL nhạy với thang đo, biến phân loại nhiều mức, đặc biệt các cột như City/Zip của Maven) | Huấn luyện không ổn định, embedding thưa | Quantile transform cho biến số; gom/loại biến phân loại cardinality cao (đã ghi ở 1.3); embedding kích thước nhỏ |
| 6 | **Ngân sách tuning không cân bằng** (GBDT: lưới 8–108 tổ hợp; DL: không gian lớn hơn nhiều) | So sánh bị chê thiên lệch | Cố định số trial Optuna và ghi rõ; cho DL cùng 5-fold CV trên train; báo thời gian tuning |
| 7 | **NODE đặc thù**: tốn bộ nhớ theo (số cây × depth), cần khởi tạo theo dữ liệu, trong bài gốc huấn luyện rất lâu | OOM, hội tụ chậm hoặc kẹt | Giảm số cây/depth; dùng batch đủ lớn cho bước khởi tạo; early stopping; chạy trên GPU Colab/Kaggle |
| 8 | **Chi phí tính toán** khi vừa Optuna vừa 5-fold vừa nhiều seed | Hết giờ Colab | Dữ liệu nhỏ nên mỗi lần chạy ngắn; chạy `n_trials` vừa phải, lưu checkpoint/kết quả trung gian; tune ít seed, chạy nhiều seed chỉ cho cấu hình tốt nhất |

**Phương án dự phòng nếu FT-T/NODE vẫn không hội tụ** (theo gợi ý trong đề cương, chọn 1–2 cái):
- **TabNet** hoặc **Mamba** (đề cương cho phép thay thế); ưu tiên mô hình ít tham số.
- **TabPFN** (mô hình được tiền huấn luyện sẵn, thiết kế cho dữ liệu bảng nhỏ cỡ vài nghìn đến ~10 nghìn dòng): làm mô hình "DL cho dữ liệu nhỏ" bổ sung. Cần kiểm tra giấy phép/cách cài đặt trước khi dùng.
- Tiền huấn luyện tự giám sát trên đặc trưng (không dùng nhãn) rồi fine-tune – chỉ làm nếu còn thời gian.
- Lưu ý: **không dùng SMOTE để "cứu" DL** ở Module 1 vì sẽ lẫn với thí nghiệm của Module 2.

**Quy trình chẩn đoán gợi ý:** (1) chạy cấu hình mặc định nhỏ → vẽ learning curve train/val; (2) nếu overfit → tăng regularization/giảm kích thước; (3) nếu không học (loss phẳng) → kiểm tra scaling, learning rate, khởi tạo; (4) so xác suất đầu ra với tỷ lệ churn thực (≈ 26–27%); (5) kiểm tra phương sai giữa seed trước khi kết luận.

> Nguồn mục "4.4. Comparing DL models" trong đề cương không nằm trong bài e-Profits mà có vẻ đến từ bài *Revisiting Deep Learning Models for Tabular Data* (Gorishniy et al.). Nhóm cần đọc bài đó và thêm vào danh mục tài liệu tham khảo.

---

## 4. GIAI ĐOẠN 2 – MODULE 2: Cân bằng nhãn bằng SMOTE

| STT | Công việc | Chi tiết | Output | Trạng thái |
|---|---|---|---|---|
| 1 | Ghi lại phân bố trước SMOTE | Số lượng và tỷ lệ Churn/Non-Churn của **tập train** IBM và Maven | Bảng "trước" | ☐ |
| 2 | Áp dụng SMOTE **chỉ trên train** | Dữ liệu trộn số + phân loại → dùng **SMOTENC** (hoặc mã hoá phù hợp), không dùng SMOTE thuần trên one-hot. Test giữ nguyên phân bố gốc | `X_train_smote`, `y_train_smote` | ☐ |
| 3 | Ghi lại phân bố sau SMOTE | Bảng và biểu đồ cột trước/sau cho cả hai dataset | Bảng + hình "trước/sau" | ☐ |
| 4 | Tránh leakage khi tuning | Nếu tune lại trong CV: đặt SMOTE bên trong `imblearn.pipeline.Pipeline` để chỉ áp lên phần train của từng fold. **Kaplan–Meier vẫn fit trên train gốc** (không phải train đã SMOTE) và `R_i`, `r_i` của validation/test lấy từ khách hàng thật | Pipeline đúng | ☐ |
| 5 | Huấn luyện lại XGBoost | Hai cấu hình: (a) **giữ nguyên siêu tham số tốt nhất** của Giai đoạn 1 (cô lập tác động của SMOTE); (b) tune lại trên dữ liệu đã SMOTE (nếu còn thời gian) | Mô hình XGB-SMOTE | ☐ |
| 6 | Dự đoán trên test | Xuất **xác suất** và **nhãn (ngưỡng 0.5)** | `m2_pred_{ibm,maven}.csv` | ☐ |
| 7 | Tính lại e-Profits | Bằng KM đã fit trên train gốc; đủ 4 biến thể (full/top20 × ARR/TRR) | `m2_eprofits.csv` | ☐ |
| 8 | So sánh trực tiếp XGB vs XGB+SMOTE | Chênh lệch **số tiền** (tuyệt đối và %); số TP/FP/FN/TN; precision/recall; tỷ lệ khách bị can thiệp; tổng chi phí can thiệp; doanh thu giữ lại; AUC, Brier | Bảng so sánh + biểu đồ phân rã e-Profits (TP − chi phí FP) | ☐ |
| 9 | Phân tích ý nghĩa | SMOTE làm gì với xác suất, ngưỡng 0.5 và e-Profits? (xem lưu ý bên dưới) | Đoạn phân tích | ☐ |
| 10 | (Tuỳ chọn) Minh hoạ "lợi nhuận ảo" | Tính e-Profits trên một tập **đã bị cân bằng** (không phải test thật) để cho thấy con số bị phình/méo so với thực tế kinh doanh – đánh dấu rõ đây là thí nghiệm minh hoạ, không hợp lệ về mặt đánh giá | Hình minh hoạ | ☐ |

**Lưu ý khi viết kết luận Module 2:**
- Đề cương nêu "SMOTE làm sai lệch hoàn toàn giá trị lợi nhuận", nhưng **bản bài báo nhóm gửi không bàn trực tiếp về SMOTE**; bài chỉ không dùng cân bằng dữ liệu và nhấn mạnh giữ phân bố gốc. Vì vậy nhóm nên **chứng minh bằng số liệu thực nghiệm** thay vì trích dẫn như một kết luận có sẵn của tác giả.
- Nếu SMOTE chỉ áp trên train và test giữ phân bố thật, thì e-Profits tính trên test **vẫn hợp lệ**. Cái bị lệch là **phân bố xác suất đầu ra** (xác suất bị đẩy lên → nhiều dự đoán churn hơn → recall tăng nhưng FP tăng) và tương tác với ngưỡng cố định 0.5. Kết quả có thể là e-Profits tăng, giảm hoặc gần như không đổi tuỳ dataset; **báo cáo đúng những gì thu được**, rồi giải thích bằng phân rã TP/FP.
- Phần "sai lệch hoàn toàn" đúng nghĩa xảy ra khi **đánh giá hoặc ước lượng KM/CLV trên dữ liệu đã cân bằng** (M2.10) – hãy tách bạch hai tình huống này trong báo cáo.

---

## 5. Ánh xạ sang cấu trúc báo cáo của môn học

| Mục báo cáo | Lấy từ |
|---|---|
| 1. Mở đầu (bối cảnh, bài gốc, động lực, đóng góp) | Tóm tắt bài e-Profits; động lực: dữ liệu nhỏ vs DL, SMOTE vs bảo toàn phân bố |
| 2. Cơ sở lý thuyết & phương pháp gốc | Eq.1–8, KM, CLV, mô hình chi phí, 6 mô hình, quy trình tuning (mục 3 bài báo) |
| 3. Phương pháp mở rộng | Module 1 (FT-T, NODE) và Module 2 (SMOTE), kèm lý do chọn |
| 4. Thực nghiệm & đánh giá | Thiết lập (split, phần cứng, siêu tham số, thang đo) → Kết quả tái hiện (1.13–1.16) → Kết quả mở rộng (M1.10, M2.8) + phân tích lỗi |
| 5. Kết luận & hướng phát triển | Tóm tắt, hạn chế (dữ liệu nhỏ, tham số giả định), hướng tiếp theo (Module 3, 4) |
| Tài liệu tham khảo | References của bài gốc + Gorishniy et al., NODE, SMOTE, Platt |

---

## 6. Sườn công việc Giai đoạn 3 (làm nếu còn thời gian)

### Module 3 – Phân tích độ nhạy kinh doanh (input: dự đoán của XGBoost)

| STT | Công việc | Chi tiết | Output |
|---|---|---|---|
| 3.1 | Quét `CPO` | Ví dụ 0.05 → 0.30 (bài báo tuyên bố thứ hạng ổn định trong [0.05, 0.2] nhưng không trình bày thí nghiệm – nhóm có thể kiểm chứng điều này). Tính e-Profits của XGB vs LightGBM (và các mô hình khác) | Biểu đồ đường e-Profits theo CPO |
| 3.2 | Điểm lỗ | Tìm CPO mà e-Profits chuyển âm; so sánh XGB vs LGBM có đổi thứ hạng không | Phân tích 1 |
| 3.3 | Kịch bản lạm phát | Quét lạm phát 0–20%: giảm `M`, tăng `c0` và `CPO` theo `(1+π)`; tìm **điểm hoà vốn** | Biểu đồ + điểm gãy |
| 3.4 | SMS vs gọi điện | Mô hình hai kênh với chi phí và tỷ lệ phản hồi khác nhau; chọn kênh theo từng phân khúc để tối đa lợi nhuận | Bảng chính sách kênh |
| 4.1 | Ablation theo cụm đặc trưng | Nhân khẩu học / dịch vụ / tài chính–hoá đơn; bỏ từng cụm, train lại XGB, đo sụt giảm e-Profits | Bảng sụt giảm tiền |
| 4.2 | Bơm nhiễu | Nhiễu 5/10/20%: Gaussian cho biến số, đảo nhãn cho biến phân loại; train lại XGB, FT-T, NODE; so đường sụt giảm Acc/AUC với e-Profits | Biểu đồ độ bền |

**Điểm cần lưu ý trước khi làm Giai đoạn 3:**
- **Lạm phát:** đề cương yêu cầu dùng hệ số/công thức có nguồn uy tín, nhưng ví dụ `M = 0.3·(1 − lạm phát)` chỉ là giả định. Cần tìm nguồn (ví dụ CPI từ Tổng cục Thống kê/World Bank và tài liệu về pass-through chi phí) để biện minh cách ánh xạ lạm phát sang `M`, `c0`, `CPO`; nếu không có thì ghi rõ đây là kịch bản giả định.
- **SMS vs gọi điện:** dữ liệu của hai bộ **không có** chi phí kênh hay tỷ lệ phản hồi; cần lấy từ tài liệu tham khảo hoặc ghi rõ là giả định.
- **Ablation:** khi bỏ cột `MonthlyCharges`/`tenure` khỏi *mô hình*, vẫn phải giữ chúng để *tính CLV/e-Profits*. Tách rõ "đặc trưng đầu vào của mô hình" và "dữ liệu dùng để tính giá trị".
- **Nhiễu:** nếu bơm nhiễu vào `MonthlyCharges`/`tenure` thì CLV cũng bị ảnh hưởng; cần quyết định nhiễu chỉ vào đầu vào mô hình hay cả đầu vào tính lợi nhuận, và ghi rõ.
- Đề cương viết "LightGCM" – đây có thể là lỗi chính tả của **LightGBM**.

---

## 7. Rủi ro & điểm cần làm rõ khi tái hiện

| # | Vấn đề | Cách xử lý |
|---|---|---|
| 1 | Bài báo không nêu tỷ lệ split, seed, `c0`/`c1` (thí nghiệm chính), cách chọn "top 20%" | Lấy từ code gốc; nếu code cũng không rõ, chọn hợp lý, ghi vào báo cáo và chạy thử độ nhạy |
| 2 | Hình 3 (Maven) có chú thích AUC giống hệt Hình 2 (IBM: rf = 0.8348…) và không khớp Bảng 4 (AUC Maven ≈ 92–94%) | Đối chiếu theo **Bảng 3–4**, không theo chú thích hình; nêu rõ trong báo cáo |
| 3 | Trục KM ghi "Account Length (Days)" trong khi văn bản nói tenure tính theo tháng | Dùng đơn vị tháng như văn bản và Eq.3 |
| 4 | Một số dòng trong Bảng 3 lặp giá trị giữa các chính sách tuning (ví dụ RF khi tune theo Accuracy và F1) và nhiều giá trị e-Profits top-20% bằng 0 | Đừng kỳ vọng khớp từng ô; so sánh theo xu hướng |
| 5 | Đề cương nói "XGBoost luôn cho e-Profits cao nhất" | Chính xác hơn: XGB cao nhất cho **toàn bộ** khi tune trên e-Profits; LightGBM/CatBoost dẫn ở **top 20%**; RF dẫn khi tune theo AUC (xem 1.16) |
| 6 | Thời gian chạy GridSearch (EBM, RF) | Dùng `n_jobs=-1`, chạy qua đêm, lưu kết quả trung gian |

---
