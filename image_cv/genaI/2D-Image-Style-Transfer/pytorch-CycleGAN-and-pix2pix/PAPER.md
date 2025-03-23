![alt text](image.png)

| 項目                  | 說明                                                                                                                                                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **論文名稱**          | Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks                                                                                                                                        |
| **發表期刊/會議**     | ICCV                                                                                                                                                                                                                   |
| **發表年份**          | 2017                                                                                                                                                                                                                   |
| **作者所屬學校/機構** | Berkeley AI Research (BAIR) laboratory, UC Berkeley                                                                                                                                                                    |
| **研究目標背景知識**  | 本研究的目標是進行圖像到圖像的轉換，特別是無需配對的訓練數據的情況下，這在很多圖像處理和計算機視覺問題中非常有用。研究利用生成對抗網絡（GAN）和循環一致性損失來達到這個目標。                                          |
| **研究目標**          | 本論文旨在學習將來源域X的圖像轉換到目標域Y的映射，即使在沒有配對示例的情況下，通過生成對抗損失來使得G(X)的圖像分布與Y的圖像分布無法區分。同時，引入逆映射F：Y→X並使用循環一致性損失來確保F(G(X)) ≈ X及G(F(Y)) ≈ Y。 |
| **資料集描述**        | 使用了多個無配對的圖像集，包括Flickr上的Monet畫作與風景照片，ImageNet中的斑馬與馬，Flickr上的夏季與冬季Yosemite照片等。                                                                                                |
| **關鍵發現**          | - 提出了CycleGAN方法，可在無配對數據的情況下實現圖像到圖像的轉換。 - 在風格轉換、物體變形、季節轉換和照片增強等多個任務上顯示出優越的效果。 - 定量比較顯示，CycleGAN優於多種現有方法。                         |
| **研究結果比較**      | 與BiGAN/ALI, CoGAN, SimGAN, 以及pix2pix等方法進行了比較，CycleGAN在多個評估指標上表現優異。                                                                                                                            |

| 項目                       | 說明                                                                                                                                                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **理論架構和設計**         | 本文提出的理論架構為CycleGAN，包括兩個映射函數G：X→Y和F：Y→X，以及相應的對抗判別器DY和DX。循環一致性損失被引入，以確保G和F的轉換是彼此的逆映射。這種架構設計在無配對的圖像到圖像轉換中顯示出很高的有效性。 |
| **模型資訊\_architecture** | CycleGAN                                                                                                                                                                                                     |
| **模型資訊\_backbone**     | ResNet                                                                                                                                                                                                       |
| **模型資訊\_參數量**       | unknown                                                                                                                                                                                                      |
| **模型資訊\_訓練成本**     | 使用Adam優化器，學習率0.0002，訓練100個epochs，然後在接下來的100個epochs中線性衰減學習率。訓練使用的硬體為NVIDIA GPU。                                                                                       |
| **模型背景知識**           | 本研究需要理解生成對抗網絡（GAN）及其應用，特別是無配對圖像翻譯和循環一致性損失的概念。                                                                                                                      |
| **模型和方法**             | CycleGAN模型包括兩個生成器G和F，以及兩個對抗判別器DY和DX。生成器的架構基於ResNet，包括6個或9個殘差塊。判別器使用70×70 PatchGAN，旨在判斷圖像塊是真實的還是生成的。                                          |
| **資料集和訓練資源**       | 使用了包括Flickr和ImageNet在內的多個資料集。訓練數據包括風景照片和畫作、斑馬和馬的圖片、以及Yosemite的季節照片。訓練在NVIDIA GPU上進行。                                                                     |
| **實驗結果比較**           | 在圖像到圖像的翻譯任務中，CycleGAN與BiGAN/ALI、CoGAN、SimGAN、pix2pix等方法進行比較，表現出更高的圖像質量和真實性。數據表明CycleGAN在多數情況下接近或超過有配對數據訓練的pix2pix方法。                       |
| **Ablation study**         | 論文通過消融實驗驗證了循環一致性損失和對抗損失對於獲得高質量結果的重要性。去除其中任何一個損失函數都會顯著降低模型性能，證明了這兩個損失函數的互補性和必要性。                                               |

# CycleGAN 技術摘要

## 資訊來源

* **Title:** Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
* **論文:**[https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.2365-6/10000000\_900554171201033\_1602411987825904100\_n.pdf?\_nc\_cat=100&ccb=1-7&\_nc\_sid=3c67a6&\_nc\_ohc=wNkFGThcmjwAb4N5Rbd&\_nc\_ht=scontent.ftpe8-2.fna&oh=00\_AfAM\_b3vi2wR6i2fuvZlon9T-WeeqovYVvd0tBr7e7\_bgA&oe=662E9827](https://scontent.ftpe8-2.fna.fbcdn.net/v/t39.2365-6/10000000_900554171201033_1602411987825904100_n.pdf?_nc_cat=100&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=wNkFGThcmjwAb4N5Rbd&_nc_ht=scontent.ftpe8-2.fna&oh=00_AfAM_b3vi2wR6i2fuvZlon9T-WeeqovYVvd0tBr7e7_bgA&oe=662E9827)
* **Arxiv:**[https://arxiv.org/abs/1703.10593](https://arxiv.org/abs/1703.10593)
* **Github:**[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

## 發表資訊

* **發表期刊/會議:** ICCV 2017
* **arxiv 發表年月:** 201703
* **作者所屬學校/機構:** Berkeley AI Research (BAIR) laboratory, UC Berkeley

## 研究貢獻

* **研究目標背景知識:** 開發一種無需配對數據的圖像到圖像轉換方法，在計算機視覺和圖像處理中具有廣泛應用。利用生成對抗網絡（GAN）和循環一致性損失實現高質量的圖像轉換。
* **研究目標:** 建立一個無需配對訓練數據即可進行圖像轉換的模型，透過生成對抗損失使得生成圖像分布與目標域圖像分布無法區分，並引入逆映射和循環一致性損失以確保映射的合理性。
* **資料集描述:** 使用了多個無配對的圖像集，包括Flickr上的Monet畫作與風景照片，ImageNet中的斑馬與馬，Flickr上的夏季與冬季Yosemite照片等。
* **關鍵發現:** CycleGAN能夠在無配對數據的情況下進行高質量的圖像到圖像轉換，應用範圍包括風格轉換、物體變形、季節轉換和照片增強等。
* **研究結果比較:** 與當前的多種基於完全監督學習的模型相比，CycleGAN在多數資料集上的表現接近或超越這些模型，特別是在無配對數據設定中表現優異。

## 模型

* **理論架構和設計:** 提出了一種基於提示的圖像轉換方法，利用生成對抗網絡和循環一致性損失實現無需配對數據的高質量圖像轉換。透過雙生成器和雙判別器架構來完成這一任務。
* **模型資訊\_architecture:** CycleGAN
* **模型資訊\_backbone:** ResNet

## 論文截錄

* **metrics & experiments:**
  
  * CycleGAN在無配對數據的情況下，在多個數據集上均顯示出接近甚至超越完全監督模型的表現。
    ![1717072136904](images/PAPER/1717072136904.png)
  * 在人類評估中，CycleGAN生成的圖像質量優於現有方法。
  * 使用前景點提示時，CycleGAN能夠生成高質量的圖像掩碼，接近手動標註的地面真相。
  * 在消融實驗中，循環一致性損失和對抗損失對於獲得高質量結果均至關重要。
    ![1717072149252](images/PAPER/1717072149252.png)
  
  原文: "To investigate this observation, we conducted an additional human study asking annotators to rate the ViTDet masks and SAM masks on the 1 to 10 quality scale used before. In Fig. 11 we observe that SAM consistently outperforms ViTDet in the human study."
* **Ablation study:**
  
  * 在不同數據量和不同訓練數據組合下，CycleGAN模型在影像轉換任務上的表現顯示出，無論是對抗損失還是循環一致性損失，都對模型性能至關重要。
  * 減少數據量會顯著降低模型性能，但使用一定量的數據（如1M影像）仍能獲得較好的效果。
  * 經過多階段累積訓練的資料引擎，每個階段均能顯著提升mIoU，顯示了累積訓練資料的有效性。

## 限制與討論

1. ​**幾何變換的挑戰**​： 當涉及顏色和紋理的變換時，該方法通常能夠成功。然而，在一些需要幾何變換的任務上卻遇到了困難。例如，對於狗轉換為貓的任務，學到的轉換往往會退化為最小化對輸入的改變。在這種情況下，我們的方法只能進行非常微小的變換，這可能是因為我們的生成器架構主要針對顏色和紋理的改變而設計。
2. ​**有配對數據與無配對數據的差距**​： 我們觀察到使用配對訓練數據的方法與我們的方法之間仍然存在一定的差距。在某些情況下，這種差距可能非常難以甚至不可能彌合。例如，我們的方法有時會在照片轉換為標籤的任務中對樹木和建築物的標籤進行錯誤的對應。解決這一問題可能需要某種形式的弱語義監督。

