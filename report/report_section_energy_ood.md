# Phần C — Energy Score và đánh giá OOD

## 1. Mục tiêu

Mô hình phân loại CIFAR-10 luôn trả về một trong 10 lớp đã học. Khi gặp ảnh ngoài phân phối, ví dụ ảnh chữ số từ SVHN, mô hình vẫn có thể gán nhãn như `cat`, `dog`, `truck`,... Vì vậy cần thêm một tầng phát hiện `Unknown` trước khi tin vào nhãn phân loại.

Phần này sử dụng **Energy-Based Out-of-Distribution Detection** để biến logits của mô hình đã train thành một điểm số OOD. Cách này không cần train thêm model mới, chỉ cần dùng output layer/logits của checkpoint đã có.

## 2. Công thức Energy Score

Với vector logits `f(x) = [f_1(x), ..., f_K(x)]`, free energy được tính như sau:

\[
E(x) = -T \log \sum_{k=1}^{K} \exp(f_k(x)/T)
\]

Trong thực nghiệm, dùng `T = 1`. Để thuận tiện cho việc đặt threshold, ta dùng **negative energy**:

\[
S(x) = -E(x) = T \log \sum_{k=1}^{K} \exp(f_k(x)/T)
\]

`S(x)` càng cao thì mẫu càng giống dữ liệu in-distribution. `S(x)` thấp nghĩa là mẫu có khả năng là OOD.

## 3. Quy tắc phát hiện Unknown

Chọn threshold `tau` dựa trên tập CIFAR-10 validation/test sao cho khoảng 95% ảnh CIFAR-10 được giữ lại là `Known`:

\[
\tau = Percentile_{5\%}(S(x_{ID}))
\]

Quy tắc quyết định:

- Nếu `S(x) > tau`: ảnh được xem là `Known`, lấy nhãn CIFAR-10 từ `argmax(logits)`.
- Nếu `S(x) <= tau`: ảnh được gán `Unknown/OOD`.

## 4. Kết quả kiểm tra hiện tại từ file CIFAR-10 logits đã có

File đầu vào:

- `cifar10_test_logits.npy`: shape `(10000, 10)`
- `cifar10_test_labels.npy`: shape `(10000,)`

Kết quả tính được:

| Chỉ số | Giá trị |
|---|---:|
| CIFAR-10 test accuracy từ logits | 92.92% |
| Mean negative energy trên CIFAR-10 | 15.9620 |
| Std negative energy trên CIFAR-10 | 6.9289 |
| Threshold `tau` tại ID TPR 95% | 4.8134 |
| Tỷ lệ CIFAR-10 bị gán Unknown tại threshold này | 5.00% |

Threshold hiện tại `tau = 4.8134` là threshold demo từ CIFAR-10 test logits. Nếu có logits validation, nên chọn threshold bằng validation rồi báo cáo kết quả cuối trên test.

## 5. Bảng kết quả cần điền sau khi chạy SVHN

Sau khi sinh `svhn_test_logits.npy`, notebook sẽ tự xuất bảng:

| Metric | Ý nghĩa |
|---|---|
| `threshold_negative_energy` | Threshold cuối dùng để phát hiện Unknown |
| `ID_TPR_known_%` | Tỷ lệ CIFAR-10 được giữ là Known, thường xấp xỉ 95% |
| `OOD_FPR95_%` | Tỷ lệ SVHN bị nhận nhầm là Known khi ID TPR = 95%; càng thấp càng tốt |
| `OOD_unknown_rate_%` | Tỷ lệ SVHN được phát hiện là Unknown; càng cao càng tốt |
| `AUROC_%` | Khả năng tách ID/OOD trên toàn bộ threshold; càng cao càng tốt |
| `AUPR_In_%` | Precision-recall nếu xem ID là class positive |

## 6. Nội dung nên đưa vào slide

- Slide 1: Vấn đề — model CIFAR-10 không nên ép ảnh SVHN vào một trong 10 lớp CIFAR.
- Slide 2: Công thức — `S(x) = logsumexp(logits)` với `T=1`.
- Slide 3: Histogram/boxplot — CIFAR-10 có negative energy cao hơn, SVHN thấp hơn.
- Slide 4: Threshold — chọn percentile 5% của CIFAR-10 để giữ 95% ID.
- Slide 5: Bảng kết quả — FPR95, AUROC, OOD Unknown rate.

## 7. Câu thuyết trình gợi ý

"Sau khi mô hình baseline trả về logits, em không dùng trực tiếp softmax confidence vì softmax có thể vẫn rất cao với ảnh ngoài phân phối. Thay vào đó, em tính negative energy bằng logsumexp của logits. Với ảnh thuộc CIFAR-10, mô hình thường tạo logits có độ lớn rõ ràng hơn nên negative energy cao. Với ảnh SVHN, negative energy có xu hướng thấp hơn. Em chọn threshold sao cho 95% ảnh CIFAR-10 vẫn được chấp nhận là Known; các ảnh có score thấp hơn threshold sẽ được gán Unknown. Như vậy pipeline cuối không chỉ phân loại 10 lớp CIFAR-10 mà còn có khả năng từ chối ảnh không thuộc phân phối huấn luyện."
