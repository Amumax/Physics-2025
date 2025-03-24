import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging
import os

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,  # можно выставить INFO или DEBUG
    format='%(levelname)s: %(message)s'
)

def thin_lens_image(
    input_image: np.ndarray,
    f: float,
    a: float,
    pixel_size: float = 1.0,
    T: float = 1.0
):
    """
    Строит изображение, формируемое тонкой линзой с фокусным расстоянием f
    при размещении предмета (input_image) на расстоянии a.
    pixel_size — условный размер пикселя (для осей x/y в одной плоскости).
    T — коэффициент пропускания (учёт потерь).
    
    Возвращает (output_image, b, M), где:
      - output_image: (H_out, W_out, [3]) массив с итоговым изображением,
      - b: расстояние от линзы до плоскости изображения (b>0 => действительное изобр.),
      - M: линейное увеличение (M<0 => переворот).
    """
    logging.info("Функция thin_lens_image запущена.")
    H_in, W_in = input_image.shape[:2]  # Если цветное, shape=(H,W,3)
    logging.debug(f"Размер входного изображения: height={H_in}, width={W_in}, ndim={input_image.ndim}")

    # Формула тонкой линзы: 1/a + 1/b = 1/f
    # => b = a*f / (a - f)
    denom = a - f
    if abs(denom) < 1e-12:
        # Ситуация, близкая к a=f => b -> ∞
        b = 1e12
        logging.warning(f"a ≈ f -> формально b -> ∞. Ставим b=1e12 (очень большое)")
    else:
        b = (a * f) / denom
    # Линейное увеличение:
    M = - b / a
    logging.debug(f"Рассчитано b={b:.3f}, M={M:.3f} (a={a}, f={f})")

    # Рассчитаем предполагаемые размеры выходной картинки
    H_out = int(abs(M) * H_in)
    W_out = int(abs(M) * W_in)
    H_out = max(H_out, 1)
    W_out = max(W_out, 1)
    logging.debug(f"Размер выходного изображения: height={H_out}, width={W_out}")

    # Создадим массив для выходного изображения
    if input_image.ndim == 3:
        output_image = np.zeros((H_out, W_out, 3), dtype=np.uint8)
    else:
        output_image = np.zeros((H_out, W_out), dtype=np.uint8)

    # Выполним «обратное отображение»
    for i_out in range(H_out):
        # координата y_out
        y_out = (i_out - H_out/2) * pixel_size
        # y_in = -(1/M)*y_out
        y_in = - (1./M) * y_out if M != 0 else 0.0
        i_in = int(H_in/2 + (y_in/pixel_size))

        for j_out in range(W_out):
            x_out = (j_out - W_out/2) * pixel_size
            x_in = - (1./M) * x_out if M != 0 else 0.0
            j_in = int(W_in/2 + (x_in/pixel_size))

            if (0 <= i_in < H_in) and (0 <= j_in < W_in):
                color = input_image[i_in, j_in]
                # Применяем коэффициент пропускания
                if input_image.ndim == 3:
                    val = (color * T).astype(np.uint8)
                else:
                    val = np.uint8(color * T)
                output_image[i_out, j_out] = val

    logging.info("Завершено построение выходного изображения.")
    return output_image, b, M


if __name__ == "__main__":
    logging.info("Запуск основного скрипта.")

    # Имя файла с входным изображением (по умолчанию 'input.jpg')
    input_filename = "input.jpg"

    # Проверяем, существует ли в текущей директории
    if not os.path.isfile(input_filename):
        logging.error(f"Файл '{input_filename}' не найден в текущей папке.")
        # Завершаем выполнение
        import sys
        sys.exit(1)
    else:
        logging.info(f"Обнаружен входной файл '{input_filename}'.")

    # Загружаем изображение
    img_in = Image.open(input_filename)
    img_in_np = np.array(img_in)
    logging.info(f"Изображение успешно загружено: {input_filename}. Shape={img_in_np.shape}")

    # Параметры тонкой линзы
    f_lens = 2.5
    a_obj  = 2.0
    T_loss = 0.9
    pixel_size = 1.0  # Условная единица размера пикселя

    logging.info(f"Параметры линзы: f={f_lens}, a={a_obj}, T={T_loss}, pixel_size={pixel_size}")

    # Строим изображение
    out_img_np, b_dist, M_factor = thin_lens_image(
        img_in_np, f_lens, a_obj, pixel_size, T_loss
    )

    if b_dist > 0:
        logging.info(f"Действительное изображение, b={b_dist:.3f} > 0")
    else:
        logging.info(f'Мнимое изображение, b={b_dist:.3f} < 0')
    logging.info(f"Увеличение M={M_factor:.3f}")

    # Преобразуем результат в PIL-изображение
    img_out = Image.fromarray(out_img_np)
    out_filename = "output_lens.jpg"
    img_out.save(out_filename)
    logging.info(f"Результат сохранён в файл '{out_filename}'.")

    # Для сравнения: простое цифровое масштабирование
    scale_factor = abs(M_factor)
    W_in, H_in = img_in.size
    W_sc = int(W_in * scale_factor)
    H_sc = int(H_in * scale_factor)
    if W_sc < 1 or H_sc < 1:
        W_sc, H_sc = 1, 1
    logging.debug(f"Простое масштабирование: W_sc={W_sc}, H_sc={H_sc}")
    img_scaled = img_in.resize((W_sc, H_sc), Image.Resampling.LANCZOS)

    # Выводим все три картинки на график
    logging.info("Формируем и показываем окно matplotlib.")
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img_in_np)
    axs[0].set_title("Исходное изображение")
    axs[0].axis("off")

    axs[1].imshow(out_img_np)
    axs[1].set_title("Через линзу (оптика)")
    axs[1].axis("off")

    axs[2].imshow(np.array(img_scaled))
    axs[2].set_title("Просто масштаб")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
    logging.info("Окно matplotlib закрыто. Завершение скрипта.")
