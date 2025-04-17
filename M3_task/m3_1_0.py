from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


H_out = 32 # считаем осью OY
W_out = 32 # считаем осью OX

class Line:
    def __init__(self, x0, y0, z0, m, n, p):
        self.offset = np.array([x0, y0, z0], dtype=float)
        self.multiply = np.array([m, n, p], dtype=float)

    def get_dot(self, t):
        return self.offset + t * self.multiply

    def intersection(self, other):
        A = np.array([self.multiply, -other.multiply, np.cross(self.multiply, other.multiply)]).T
        B = np.array(other.offset) - np.array(self.offset)
        try:
            t, s, _ = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return None
        
        return self.get_dot(t)
    
def make_lines_two_points(point_1, point_2):
    return Line(*point_1, *(point_2 - point_1))
    
class Plane:
    def __init__(self, A, B, C, D):
        self.normal_vector = np.array([A, B, C])
        self.free = D

    def intersection_with_line(self, line):
        t = -(np.dot(self.normal_vector, line.offset) + self.free) / np.dot(self.normal_vector, line.multiply)
        return line.get_dot(t)
    

def thin_lens_image_input_lines(
    input_image: np.ndarray,
    f: float,
    a: float,
):
    H_in, W_in = input_image.shape[:2]
    error = [0]
    try:
        b = (a * f) / (a - f)
    except ZeroDivisionError:
        b = 1e9 # типо бесконечность 
    

    M = b / - a   # Увеличение изображение
                # Если M > 0 то     перевернутое    действительное
                # Если M < 0 то     прямое          мнимое

    
    output_image = np.zeros((H_out, W_out, 3), dtype=np.uint8)


    # отдельно считаем для центра картинки 
    color = input_image[H_in // 2, W_in // 2]
    val = (color).astype(np.uint8)
    output_image[int(np.round(H_out / 2)), int(np.round(W_out / 2))] = val

    for i_in in range(H_in):
        for j_in in range(W_in):
            y_in = i_in - H_in / 2
            x_in = j_in - W_in / 2
            # рассмотрим ход лучей в плоскости 
            #                           точка на картине
            #                           главная оптическа ось 

            # луч 1 проходит через точку и центр линзы и не преломляется
            # луч 2 проходит параллельно оптической оси линзы
            #   и преломлятся в фокус

            l1 = make_lines_two_points(np.array([0, 0, 0]), np.array([x_in, y_in, -a]))
            l2 = make_lines_two_points(np.array([0, 0, f]), np.array([x_in, y_in, 0]))

            out_point = l1.intersection(l2)
            if out_point is None:
                # print("Паралельные лучи из точки: ", i_in, j_in)
                continue
            error.append(out_point[2] - b)
            
            i_out, j_out = int(np.round(out_point[1] + H_out / 2)),  int(np.round(out_point[0] + W_out / 2))

            if 0 <= i_out < H_out and 0 <= j_out < W_out:
                color = input_image[i_in, j_in]

                val = (color).astype(np.uint8)
                output_image[i_out, j_out] = val
    
    return output_image, b, M, sum(error)/len(error)


def thin_lens_image_output_lines(
    input_image: np.ndarray,
    f: float,
    a: float
):
    H_in, W_in = input_image.shape[:2]
    error = [0]
    try:
        b = (a * f) / (a - f)
    except ZeroDivisionError:
        b = 50 # типо бесконечность 
    

    M = b / - a   # Увеличение изображение
                # Если M > 0 то     перевернутое    действительное
                # Если M < 0 то     прямое          мнимое
    
    
    output_image = np.zeros((H_out, W_out, 3), dtype=np.uint8)

    # отдельно считаем для центра картинки 
    color = input_image[H_in // 2, W_in // 2]
    val = (color).astype(np.uint8)
    output_image[int(np.round(H_out / 2)), int(np.round(W_out / 2))] = val

    for i_out in range(H_out):
        for j_out in range(W_out):
            y_out = i_out - H_out / 2
            x_out = j_out - W_out / 2
            # рассмотрим ход лучей в плоскости 
            #                           точка на картине
            #                           главная оптическа ось 

            # луч 1 проходит через точку и центр линзы и не преломляется
            # луч 2 проходит параллельно оптической оси линзы
            #   и преломлятся в фокус

            l1 = make_lines_two_points(np.array([0, 0, 0]), np.array([x_out, y_out, b]))
            l2 = make_lines_two_points(np.array([0, 0, -f]), np.array([x_out, y_out, 0]))
            inp_img_plane = Plane(0, 0, 1, a) # точки на плоскости картины

            in_point_1 = inp_img_plane.intersection_with_line(l1)
            in_point_2 = inp_img_plane.intersection_with_line(l2)
            error.append(in_point_1 - in_point_2)
            in_point = (in_point_1 + in_point_2) / 2
            if in_point is None:
                # print("Паралельные лучи из точки: ", i_out, j_out)
                continue
            # error.append(in_point[2] - a)
            
            i_in, j_in = int(np.round(in_point[1] + H_in / 2)),  int(np.round(in_point[0] + W_in / 2))

            if 0 <= i_in < H_in and 0 <= j_in < W_in:
                color = input_image[i_in, j_in]

                val = (color).astype(np.uint8)
                output_image[i_out, j_out] = val
    
    return output_image, b, M, sum(error)/len(error)


if __name__ == "__main__":
    img_in = Image.open("/Users/lida-os/mipt/physic/Physics-2025/M3_task/pixel_art.jpg")
    img_in_np = np.array(img_in) 
        
    # Параметры тонкой линзы
    f_lens = 8   
    a_obj = 2
        
        # Запуск преобразования
    out, b_dist, M_factor, error = thin_lens_image_input_lines(
        img_in_np, f_lens, a_obj
    )
    
    another_out, _, _, error_2 = thin_lens_image_output_lines(
        img_in_np, f_lens, a_obj
    )

    if b_dist > 0:
        print(f"Действительное изображение, b={b_dist:.3f} > 0")
    else:
        print(f'Мнимое изображение, b={b_dist:.3f} < 0')
    print(f"Увеличение M={M_factor:.3f}")
    print(f'Ошибка в расчете координаты итоговой картинки 1 = {error}')
    print(f'Ошибка в расчете координаты итоговой картинки 2 = {np.sqrt(sum(i*i for i in list(error_2))) / len(error_2)}')

    img_out = Image.fromarray(out)
    img_out.save("/Users/lida-os/mipt/physic/Physics-2025/M3_task/output_lens.jpg")

    img_out_another = Image.fromarray(another_out)
    img_out_another.save("/Users/lida-os/mipt/physic/Physics-2025/M3_task/output_lens_another.jpg")
        

    # Попытка сделать микроскоп 


    f1, f2 = 2, 4
    L = 10
    a = 3

    H_out = 500 # считаем осью OY
    W_out = 500 # считаем осью OX


    # mic_out_1, b_1, M_mic_1, error_mic = thin_lens_image_input_lines(
    #     img_in_np, f1, a
    # )
    # делаем красиво
    mic_out_1, b_1, M_mic_1, error_mic = thin_lens_image_output_lines(
        img_in_np, f1, a
    )
    
    if b_1 > 0:
        print(f"Действительное изображение, b={b_1:.3f} > 0")
    else:
        print(f'Мнимое изображение, b={b_1:.3f} < 0')

    mic_out_2, b_2, M_mic_2, error_mic = thin_lens_image_output_lines(
        mic_out_1, f2, L-b_1
    )
    if b_2 > 0:
        print(f"Действительное изображение, b={b_2:.3f} > 0")
    else:
        print(f'Мнимое изображение, b={b_2:.3f} < 0')
    print(f"Увеличение M1={M_mic_1:.3f}, M2={M_mic_2:.3f}")
    print(f'Ошибка в расчете координаты итоговой картинки 1 = {error_mic}')


    # Покажем всё на matplotlib:
    fig, axs = plt.subplots(2, 3, figsize=(12, 4), facecolor='lightgray')
    axs[0, 0].imshow(img_in_np)
    axs[0, 0].set_title("Исходное изображение")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(img_out)
    axs[0, 1].set_title("После тонкой линзы")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(img_out_another)
    axs[0, 2].set_title("После тонкой линзы")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(mic_out_1)
    axs[1, 0].set_title("После микроскопа середина")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(mic_out_2)
    axs[1, 1].set_title("После микроскопа финал")
    axs[1, 1].axis("off")

    plt.tight_layout()
    plt.show()
