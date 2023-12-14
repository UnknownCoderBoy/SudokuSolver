
import cv2
import numpy as np
import pandas as pd
import time
import joblib
from skimage.feature import hog as skhog
import os
import copy
import random
import streamlit as st

def preprocess(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray


def extract_frame(img):
    mask = np.zeros(img.shape, np.uint8)

    thresh = cv2.adaptiveThreshold(img, 255, 0, 1, 9, 5)
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = []
    res = []
    max_value = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        peri = cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, 0.01 * peri, True)
        if (len(approximated) == 4) and (area > max_value) and (area > 40000):
            max_value = area
            biggest_contour = approximated
    if len(biggest_contour) > 0:
        cv2.drawContours(mask, [biggest_contour], 0, 255, -1)
        cv2.drawContours(mask, [biggest_contour], 0, 0, 2)
        res = cv2.bitwise_and(img, mask)
    return res, biggest_contour, mask, thresh


def get_corners(contour):
    largest_contour = contour.reshape(len(contour), 2)
    sum_vectors = largest_contour.sum(1)
    sum_vectors_2 = np.delete(largest_contour, [np.argmax(sum_vectors), np.argmin(sum_vectors)], 0)

    corners = np.float32([
        largest_contour[np.argmin(sum_vectors)],
        sum_vectors_2[np.argmax(sum_vectors_2[:, 0])],
        sum_vectors_2[np.argmin(sum_vectors_2[:, 0])],
        largest_contour[np.argmax(sum_vectors)]
    ])
    return corners


def perspective_transform(img, shape, corners):
    pts2 = np.float32(
        [[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])  # Apply Perspective Transform Algorithm

    matrix = cv2.getPerspectiveTransform(corners, pts2)
    result = cv2.warpPerspective(img, matrix, (shape[0], shape[1]))

    return result


def extract_numbers(img):
    result = preprocess_numbers(img)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(result)

    viz = np.zeros_like(result, np.uint8)

    centroidy = []
    stats_numbers = []

    for i, stat in enumerate(stats):
        if i == 0:
            continue
        if stat[4] > area and stat[2] in range(5,40) and stat[3] in range(5,40) and stat[0] > 0 and stat[
            1] > 0 and (int(stat[3] / stat[2])) in range(1,5):
            viz[labels == i] = 255
            centroidy.append(centroids[i])
            stats_numbers.append(stat)

    stats_numbers = np.array(stats_numbers)
    centroidy = np.array(centroidy)
    return viz, stats_numbers, centroidy


def preprocess_numbers(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img


def center_numbers(img, stats, centroids):
    centered_num_grid = np.zeros_like(img, np.uint8)
    matrix_mask = np.zeros((9, 9), dtype='uint8')
    for i, number in enumerate(stats):
        left, top, width, height, area = stats[i]
        img_left = int(((left // 50)) * 50 + ((50 - width) / 2))
        img_top = int(((top // 50)) * 50 + ((50 - height) / 2))
        center = centroids[i]

        centered_num_grid[img_top:img_top + height,
        img_left: img_left + width] = img[number[1]:number[1] + number[3],
                                                                 number[0]:number[0] + number[2]]
        y = int(np.round((center[0] + 5) / 50, 1))
        x = int(np.round((center[1] + 5) / 50, 1))
        matrix_mask[x, y] = 1
    return centered_num_grid, matrix_mask


def procces_cell(img):
    cropped_img = img[5:img.shape[0] - 5, 5:img.shape[0] - 5]
    if selected_model == "Custom":
            resized = cv2.resize(cropped_img, (40, 40))
    elif selected_model == "MINIST":
            resized = cv2.resize(cropped_img, (28, 28))
    return resized


def displayNumbers(img, numbers, solved_num, color=(0, 255, 0)):
    grid = np.zeros((9, 9), dtype=int)
    w = int(img.shape[1] / 9)
    h = int(img.shape[0] / 9)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(9):
        for j in range(9):
            if numbers[j, i] == 0:
                grid[j,i] = solved_num[j, i]
                cv2.putText(img, str(solved_num[j, i]),
                            (i * w + int(w / 2) - int((w / 4)), int((j + 0.7) * h)),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color,
                            1, cv2.LINE_AA)
    return img, grid

def predNumbers(img, numbers, color=(0, 0, 255)):
    w = int(img.shape[1] / 9)
    h = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[j, i] != 0:
                cv2.putText(img, str(numbers[j, i]),
                            (i * w + int(w / 2) - int((w / 4)) + 20, int((j + 0.4) * h)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, color,
                            1, cv2.LINE_AA)
    return img

def get_inv_perspective(img, img_solved, corners, height=450, width=450):

    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([corners[0], corners[1], corners[2], corners[3]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img_solved, matrix, (img.shape[1],
                                                      img.shape[0]))
    return result


def inv_transformation(mask,img,predicted_matrix,solved_matrix,corners):
    img_solved, grid = displayNumbers(mask, predicted_matrix, solved_matrix)
    inv = get_inv_perspective(img, img_solved, corners)
    img = cv2.addWeighted(img,1, inv,1, 0,-1)
    return img, img_solved, inv, grid

def findEmpty(board):
    for y, row in enumerate(board):
        for x, val in enumerate(row):
            if val == 0:
                return y, x  # y = row, x = column
    return None


def validCheck(board, number, coordinates):
    row, col = coordinates

    # Check row and column
    if number in board[row] or number in [board[i][col] for i in range(9)]:
        return False

    # Check the 3x3 box
    box_row, box_col = row // 3 * 3, col // 3 * 3
    for i in range(box_row, box_row + 3):
        for j in range(box_col, box_col + 3):
            if board[i][j] == number:
                return False

    return True


def generateRandomBoard(board):
    find = findEmpty(board)
    if find is None:
        return True

    row, col = find
    valid_numbers = list(range(1, 10))
    random.shuffle(valid_numbers)

    for number in valid_numbers:
        if validCheck(board, number, (row, col)):
            board[row][col] = number
            if generateRandomBoard(board):
                return True

            board[row][col] = 0
    return False


def deleteCells(firstBoard, number):
    while number:
        row = random.randint(0, 8)
        col = random.randint(0, 8)
        if firstBoard[row][col] != 0:
            firstBoard[row][col] = 0
            number = number - 1


def sudokuGenerate(level):
    firstBoard = np.zeros((9, 9), dtype=int)

    generateRandomBoard(firstBoard)
    if level == 1:
        deleteCells(firstBoard, 30)
    if level == 2:
        deleteCells(firstBoard, 40)
    if level == 3:
        deleteCells(firstBoard, 50)
    return firstBoard


def solveSudoku(board):
    find = findEmpty(board)
    if not find:
        return True
    else:
        row, col = find

    for number in range(1, 10):
        if validCheck(board, number, (row, col)):
            board[row][col] = number
            if solveSudoku(board):
                return True
            board[row][col] = 0

    return False


def solve_sudoku(partially_solved_sudoku):
    solved_sudoku = copy.deepcopy(partially_solved_sudoku)
    if solveSudoku(solved_sudoku):
        return solved_sudoku
    else:
        return np.zeros((9, 9), dtype=int)

def find_incorrect_rows_columns_subgrids(matrix):
    incorrect_rows = []
    incorrect_columns = []
    incorrect_subgrids = []

    for i in range(9):
        row = matrix[i, :]
        col = matrix[:, i]
        subgrid_row = (i // 3) * 3
        subgrid_col = (i % 3) * 3
        subgrid = matrix[subgrid_row:subgrid_row + 3, subgrid_col:subgrid_col + 3]

        # Remove 0 values before checking for duplicates
        row_without_zeros = row[row != 0]
        col_without_zeros = col[col != 0]
        subgrid_without_zeros = subgrid[subgrid != 0]

        if len(np.unique(row_without_zeros)) != len(row_without_zeros):
            incorrect_rows.append(i)
        if len(np.unique(col_without_zeros)) != len(col_without_zeros):
            incorrect_columns.append(i)
        if len(np.unique(subgrid_without_zeros)) != len(subgrid_without_zeros):
            incorrect_subgrids.append((subgrid_row, subgrid_col))

    return incorrect_rows, incorrect_columns, incorrect_subgrids


def predict_numbers(numbers, matice):

    if selected_model == "Custom":
        model_path = 'digit_recognition_model.xml'
        svm = cv2.ml.SVM_load(model_path)
    elif selected_model == "MINIST":
        model_path = 'minist_digit_recognition_model.xml'  # Update with the path to your other model
        svm = cv2.ml.SVM_load(model_path)

    winSize = (28,28)
    blockSize = (14, 14)
    blockStride = (7, 7)
    cellSize = (7, 7)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # Load the trained SVM model
    # clf = joblib.load("digit_recognition_model.pkl")  # Replace with the path to your trained model file


    for row in range(9):
        for col in range(9):
            if matice[row, col] == 1:
                cell = numbers[50 * row: (50 * row) + 50, 50 * col: (50 * col) + 50]
                cell = procces_cell(cell)

                # Calculate HOG features for the test image
                hog_feature = hog.compute(cell)
                # Use the trained SVM model to predict the digit
                _, predicted_digit = svm.predict(hog_feature.reshape(1, -1))

                # Calculate HOG features for the test image
                # hog_features = skhog(cell, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)
                # Use the trained SVM model to predict the digit
                # predicted_digit = clf.predict(hog_features)

                # cv2_imshow(cell)
                # print(int(predicted_digit[0][0]))
                matice[row, col] = int(predicted_digit[0][0])
    return matice


def style_rows(row):
    if row.name in [2, 5]:
      return ['border-bottom: 4px solid'] * len(row)
    return [''] * len(row)

def style_columns(col):
    if col.name in [3, 6]:
      return ['border-left: 4px solid'] * len(col)
    return [''] * len(col)

def display_table(grid, label):
	df = pd.DataFrame(grid)
	styled_df_rows = df.style.apply(style_rows, axis=1)
	styled_df_columns = df.style.apply(style_columns, axis=0)
	styled_df_combined = styled_df_rows.use(styled_df_columns.export())

	st.markdown("""
      <style>
          table {
              text-align: center;
          }
          button[title="View fullscreen"]{
              visibility: hidden;
          }
      </style>
  """, unsafe_allow_html=True)
	html = styled_df_combined.hide_index().hide_columns().render()
	#html = '<div style="display: flex; justify-content: center;">'+html+'</div>'
	html = html + '<div class="st-emotion-cache-ltfnpr" style="width: 300px; padding-bottom: 30px;">'+label+'</div>'
	st.write(html, unsafe_allow_html=True)

st.set_page_config(page_title="Sudoku Solver")


with st.sidebar:
	# sel = st.selectbox(
  #  "How would you like to be solve?",
  #  ("Automatically Solve", "Detect & Solve Manually"))
  st.subheader("Sudoku Solver")
  selected_model = st.sidebar.radio("Select Model", ["Custom", "MINIST"])
  st.write("")
  area= st.slider('Select area', 50, 300, 80, 10)



st.title("Sudoku Solver")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

	with st.spinner('Wait Processing Image...'):
		time.sleep(5)

	img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
	og = img.copy()

	gray_img = preprocess(img)
	frame, contour, contour_line, thresh = extract_frame(gray_img)
	if contour is not None:
		st.success("Done")
		img_with_contour = img.copy()
		cv2.drawContours(img_with_contour, [contour], 0, (0, 255, 0), 3)

		corners = get_corners(contour)
		perstrans = perspective_transform(img, (450, 450), corners)
		result = perspective_transform(frame, (450, 450), corners)

		img_nums, stats, centroids = extract_numbers(result)
		centered_numbers, matrix_mask = center_numbers(img_nums, stats, centroids)
		predicted_matrix = predict_numbers(centered_numbers, matrix_mask)
		predOverlay_image = predNumbers(perstrans.copy(), predicted_matrix)
		predicted_matrix = st.sidebar.data_editor(predicted_matrix, hide_index=False)
		incorrect_rows, incorrect_columns, incorrect_subgrids = find_incorrect_rows_columns_subgrids(predicted_matrix)
		solved_matrix = solve_sudoku(predicted_matrix.copy())
		mask = np.zeros_like(result)
		img, img_solved, inv, solved_grid = inv_transformation(mask, img, predicted_matrix, solved_matrix, corners)


		col1, col2 = st.columns(2)
		with col1:
			st.image(og, channels="BGR", caption="Uploaded Image", width=300)
			st.image(perstrans, channels="BGR", caption="Perspective Transform", width=300)
			st.image(centered_numbers, caption="Centered Numbers for Recognition", width=300)
			display_table(predicted_matrix, "Predicted Grid")


		with col2:
			st.image(img_with_contour, channels="BGR", caption="Sudoku Frame", width=300)
			st.image(img_nums, caption="Get Numbers (Connected Components)", width=300)
			st.image(predOverlay_image, channels="BGR", caption="Predicted Digit", width=300)
			display_table(solved_grid, "Solved Grid")

		columns = st.columns(3)
		if incorrect_rows:
				with columns[0]:
					st.text("Incorrect Rows:")
					for row in incorrect_rows:
						st.write(f"Row Index {row}")

		if incorrect_columns:
				with columns[1]:
					st.text("Incorrect Column:")
					for col in incorrect_columns:
						st.write(f"Column Index {col}")

		if incorrect_subgrids:
				with columns[2]:
					st.text("Incorrect Subgrids:")
					for subgrid in incorrect_subgrids:
						st.write(f"Subgrid at (row {subgrid[0]//3}, col {subgrid[1]//3})")

		if (incorrect_rows or incorrect_columns or incorrect_subgrids):
			st.write("")
			st.write("")

		col1, col2 = st.columns(2)
		with col1:
			st.image(img_solved, caption="Inverse Perspective", width=300)

		with col2:
			st.image(inv, channels="BGR", caption="Solved Sudoku", width=300)
		st.image(img, channels="BGR", caption="Overlayed Solution", width=300)

	else:
		st.error("Failed to Process")
