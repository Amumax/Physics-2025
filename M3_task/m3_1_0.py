import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def thin_lens_image(
    input_image: np.ndarray,
    f: float,
    a: float,
    pixel_size: float = 1.0,
    T: float = 1.0
):
    """
    Строит изображение, формируемое тонкой линзой с фокусным расстоянием f
    при размещении предмета (input_image) на расстоянии a (центр линзы берем за x=0).
    pixel_size — условный размер пикселя по оси y и x (для масштаба).
    T — коэффициент пропускания (учет потерь).
    
    Возвращает (output_image, b, scale_out), где:
    - output_image: 2D (или 3D) массив с итоговым изображением,
    - b: рассчитанное расстояние от линзы до экрана/изображения (может быть < 0, если изображение мнимое),
    - scale_out: масштаб (сколько пикселей входного изображения приходится на пиксель выходного).
    """
    # Размер входного изображения
    H_in, W_in = input_image.shape[:2]  # если цветное, shape=(H,W,3)
    
    # Уравнение тонкой линзы: 1/a + 1/b = 1/f
    # -> b = (a * f) / (a - f)
    # Если a < f, то b < 0 => мнимое изображение (по эту же сторону линзы).
    # Если a > f, то b > 0 => действительное изображение (с другой стороны).
    # Для a=f → b → ∞ (либо "глаз на бесконечность")
    
    if abs(a - f) < 1e-9:
        # Ситуация a ~ f: формально b -> бесконечность.
        # Для наглядности берем большое b
        b = 1e9
    else:
        b = (a * f) / (a - f)
    
    # Линейное увеличение M = -b / a (знак показывает перевернутость)
    M = - b / a
    
    # Определим желаемый размер выходного изображения
    # Пусть вых. картинка ограничена размерами H_out, W_out = int(abs(M)*H_in), int(abs(M)*W_in).
    H_out = int(abs(M) * H_in)
    W_out = int(abs(M) * W_in)
    if H_out < 1 or W_out < 1:
        H_out, W_out = 1, 1
    
    # Создадим массив для результата
    # Если входное изображение 3-канальное (RGB), исходное shape = (H_in, W_in, 3)
    # Иначе 2D (однотонное). Чтобы универсально, проверим ndim:
    if input_image.ndim == 3:
        output_image = np.zeros((H_out, W_out, 3), dtype=np.uint8)
    else:
        output_image = np.zeros((H_out, W_out), dtype=np.uint8)
    
    # Для упрощения считаем, что:
    # - предмет лежит на плоскости x = -a (ось X направлена "горизонтально"),
    # - линза в плоскости x=0,
    # - экран (или виртуальная плоскость) в x = +b (может быть < 0, если мнимое изобр.)
    # - ось Y перпендикулярна оптической оси, с пикс. размером pixel_size.
    
    # Центр предмета = (x=-a, y=0). Соответственно:
    #   пиксел (i_in, j_in) => y_in = (i_in - H_in/2)*pixel_size, ...
    # Аналогично для выходного изображения (i_out, j_out).
    
    # Для каждого пикселя выходного изображения (i_out, j_out) определим,
    # откуда в предмете пришел соответствующий луч (i_in, j_in).
    #
    # Параксиальная оптика (малые углы): используем соотношения подобия
    #   y_out / (b) = y_in / (a),  =>  y_in = y_out * (a / b).
    # По вертикали: y_out = (i_out - H_out/2)*pixel_size.
    # => i_in = H_in/2 + (y_in / pixel_size).
    #
    # Аналогично по горизонтали. Но тут предположим:
    #   x-координаты напрямую заданы a,b. Если предмет размер W_in,
    #   то "горизонтальную" координату (j_in) тоже масштабируем тем же M.
    #
    # Здесь, для простоты, «плоскость предмета» и «экран» считаем параллельными,
    #   т.е. вертикальные оси совпадают, горизонтальные оси совпадают.
    
    # Коэффициент для перевода y_out -> y_in:
    #   y_in = (a / b) * y_out.
    # Но M = -b / a => a/b = -1/M
    # => y_in = - (1/M) * y_out
    # Также по горизонтали x_in = - (1/M) * x_out
    
    # Потери интенсивности T (например, T~0.9...0.95 если хотим учесть отражения на 2 границах)
    
    # Выполним «обратное отображение»:
    for i_out in range(H_out):
        # координата y_out
        y_out = (i_out - H_out/2) * pixel_size
        # Находим соответствующую y_in
        y_in = - (1.0/M) * y_out  # y_in = (a/b)*y_out, но M= - b/a => a/b= -1/M
        # индекс i_in в пикселях:
        i_in = int(H_in/2 + (y_in / pixel_size))
        
        # По горизонтали
        for j_out in range(W_out):
            x_out = (j_out - W_out/2) * pixel_size
            x_in = - (1.0/M) * x_out
            j_in = int(W_in/2 + (x_in / pixel_size))
            
            # Если (i_in, j_in) в пределах исходного изображения, то берём цвет
            if 0 <= i_in < H_in and 0 <= j_in < W_in:
                color = input_image[i_in, j_in]
                # Учтем потери (T)
                if color.ndim == 0:  # ч/б
                    val = int(color * T)
                    output_image[i_out, j_out] = val
                else:
                    # RGB
                    val = (color * T).astype(np.uint8)
                    output_image[i_out, j_out] = val
            else:
                # Иначе фон (чёрный)
                pass
    
    return output_image, b, M


if __name__ == "__main__":
    # Пример использования
    # Загрузим картинку "input.jpg" из текущей директории (не забудьте поместить её туда).
    img_in = Image.open("input.jpg")
    img_in_np = np.array(img_in)  # numpy массив
    
    # Параметры лупы (тонкой линзы)
    f_lens = 2.5   # фокусное расстояние, условные единицы
    a_obj = 2.0    # расстояние предмета до линзы, условные единицы
    T_loss = 0.95  # Коэффициент пропускания
    
    # Запуск преобразования
    out, b_dist, M_factor = thin_lens_image(
        img_in_np, f_lens, a_obj,
        pixel_size=1.0,
        T=T_loss
    )
    
    print(f"Тонкая линза: f={f_lens}, a={a_obj}, b={b_dist:.2f}, M={M_factor:.2f}")
    if b_dist > 0:
        print("Действительное изображение (b > 0), перевёрнутое (M < 0 => инвертируется по вертикали).")
    else:
        print("Мнимое изображение (b < 0): находится по ту же сторону линзы, увеличение |M|.")
    
    # Преобразуем массив обратно в PIL-изображение и сохраним
    img_out = Image.fromarray(out)
    img_out.save("output_lens.jpg")
    
    # Сравним с простым скейлом (увеличением) в тот же размер:
    # Возьмем масштаб по модулю |M|:
    scale_factor = abs(M_factor)
    W_in, H_in = img_in.size
    W_sc = int(W_in * scale_factor)
    H_sc = int(H_in * scale_factor)
    img_scaled = img_in.resize((W_sc, H_sc), Image.Resampling.LANCZOS)
    
    # Покажем всё на matplotlib:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_in_np)
    axs[0].set_title("Исходное изображение")
    axs[0].axis("off")
    
    axs[1].imshow(img_out)
    axs[1].set_title("После тонкой линзы")
    axs[1].axis("off")
    
    axs[2].imshow(img_scaled)
    axs[2].set_title("Просто увеличение")
    axs[2].axis("off")
    
    plt.tight_layout()
    plt.show()
