# Import Necessary Modules
from scipy.stats import entropy
from filter_feature_points import check_point_in_bbox

# Import Necessary Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time



def draw_features_on_image_vertical(img_1, img_2, img1_pts, img2_pts, pic_title):
        
        radius = 5
        color = (0, 255, 0)
        thickness = 2

        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()


        for i, (x, y) in enumerate(img1_pts):
            cv2.circle(img_1_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x, y) in enumerate(img2_pts):
            cv2.circle(img_2_copy, (int(x), int(y)), radius, color, thickness)


        fig = plt.figure(figsize=(10, 5))

        plt.title(pic_title)
        plt.imshow(np.vstack((img_1_copy, img_2_copy)))

        for (x1, y1), (x2, y2) in zip(img1_pts, img2_pts):
            plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]], 'b', linewidth=1)
        

        return fig





def draw_features_on_image_together(img_1, img_2, img1_pts, img2_pts, pic_title):
    
    radius = 5
    thickness = 2

    img_1_copy = img_1.copy()


    for i, (x, y) in enumerate(img1_pts):
        cv2.circle(img_1_copy, (int(x), int(y)), radius, (0, 255, 0), thickness)

    for i, (x, y) in enumerate(img2_pts):
        cv2.circle(img_1_copy, (int(x), int(y)), radius, (255, 0, 0), thickness)


    fig = plt.figure()
    plt.title(pic_title)
    plt.imshow(img_1_copy)
    plt.title(pic_title)

    # for (x1, y1), (x2, y2) in zip(img1_pts, img2_pts):
    #     plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]],              'b', linewidth=0.5)
    

    plt.show()





def visualize_KL(img_1, img_2, img1_pts, img2_pts, kl_values, left_boxes, right_boxes, font_size, pic_title):
        
        radius = 2
        color = (0, 255, 0)
        thickness = 2

        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()




        for i, (x, y) in enumerate(img1_pts):
            cv2.circle(img_1_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x1, y1, x2, y2) in enumerate(left_boxes):
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)
            cv2.rectangle(img_1_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for i, (x1, y1, x2, y2) in enumerate(right_boxes):
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)
            cv2.rectangle(img_1_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for i, (x, y) in enumerate(img2_pts):
            cv2.circle(img_2_copy, (int(x), int(y)), radius, color, thickness)

        for i, (x1, y1, x2, y2) in enumerate(left_boxes):
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)
            cv2.rectangle(img_2_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

        for i, (x1, y1, x2, y2) in enumerate(right_boxes):
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)
            cv2.rectangle(img_2_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)


        fig = plt.figure()

        plt.title(pic_title)
        plt.imshow(np.vstack((img_1_copy, img_2_copy)))

        for (x1, y1), (x2, y2), one_kl_value in zip(img1_pts, img2_pts, kl_values):
            plt.plot([x1, x2], [y1, y2+img_1_copy.shape[0]], 'b', linewidth=0.5)
            plt.text(x1, y1, one_kl_value, fontsize=font_size, color='red', rotation=45)
        

        plt.pause(1)
        plt.close


def visualize_KL_result(img_1, img_2, img1_pts, img2_pts, kl_values, left_boxes, right_boxes, pic_title):
        

        img_1_copy = img_1.copy()
        img_2_copy = img_2.copy()

        fig = plt.figure()
        
        lines = []
        color = np.random.randint(0, 256, (len(left_boxes), 3))
        color_normalized = color/255.0

        for (x1, y1, x2, y2), one_color in zip(left_boxes, color_normalized):
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)          

            kl_values_in_one_bb = []
            for (x, y), one_kl_value in zip(img2_pts, kl_values):
                 if check_point_in_bbox((x1, y1, x2, y2), (x, y)):
                      kl_values_in_one_bb.append(one_kl_value)

            # print(np.mean(kl_values_in_one_bb))
            label = f"{(x1, y1, x2, y2)}"
            lines.append({'y':np.mean(kl_values_in_one_bb), 'color':one_color, 'label':label})        

        # mean_labels = [m.get_label() for m in mean_lines]
        # plt.legend(mean_lines, mean_labels)
        fig, axs = plt.subplots(2, 1)

        # Add horizontal lines
        for line in lines:
            axs[0].axhline(y=line['y'], color=line['color'], label=line['label'])

        # Add legend
        axs[0].legend()


        radius = 2
        dot_color = (0, 255, 0)
        thickness = 2

        img_1_copy = img_1.copy()

        for i, (x, y) in enumerate(img2_pts):
            cv2.circle(img_1_copy, (int(x), int(y)), radius, dot_color, thickness)

        for (x1, y1, x2, y2), one_color in zip(left_boxes, color):
            one_color = one_color.tolist()
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)
            image_1_result = cv2.rectangle(img_1_copy, (x1, y1), (x2, y2), one_color, 5)

        for (x1, y1, x2, y2), one_color in zip(right_boxes, color):
            one_color = one_color.tolist()
            increase_x_scale = round((x2-x1)*0.2)
            increase_y_scale = round((y2-y1)*0.2)
            image_1_result = cv2.rectangle(img_1_copy, (x1, y1), (x2, y2), one_color, 5)

        axs[1].imshow(image_1_result)
    
        # Show plot

        plt.pause(1)
        plt.close
