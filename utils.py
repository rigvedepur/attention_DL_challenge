import cv2

def slope(pt1, pt2):
    (x1,y1) = pt1
    (x2, y2) = pt2
    return (y2-y1)/((x2-x1)+0.001)

def plot_skeletal_map(frame, pt_indices, points, message, text_color, stats, t_elapsed):

    attentive, distracted, absent = stats

    total = attentive + distracted + absent
    attentive_percent = round(attentive * 100 / total, 2)
    distracted_percent = round(distracted * 100 / total, 2)
    absent_percent = round(absent * 100 / total, 2)

    if t_elapsed > 60:
        session_time = str(int(t_elapsed / 60)) + 'min ' + str(t_elapsed % 60) + 'sec'
    else:
        session_time = str(t_elapsed) + ' seconds'

    cv2.putText(frame, 'Session time: ' + session_time, (450, 25), cv2.FONT_HERSHEY_SIMPLEX, .8,
                (50, 50, 50), 1, lineType=cv2.LINE_AA)

    cv2.putText(frame, 'Attentive: ' + str(attentive_percent) + '%', (550, 60), cv2.FONT_HERSHEY_SIMPLEX, .8,
                (0, 150, 50), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, 'Distracted: ' + str(distracted_percent) + '%', (550, 85), cv2.FONT_HERSHEY_SIMPLEX, .8,
                (0, 0, 180), 1, lineType=cv2.LINE_AA)
    cv2.putText(frame, 'Absent: ' + str(absent_percent) + '%', (550, 110), cv2.FONT_HERSHEY_SIMPLEX, .8,
                (200, 0, 0), 1, lineType=cv2.LINE_AA)

    if len(pt_indices) == 0:
        cv2.putText(frame, message, (30, 25), cv2.FONT_HERSHEY_COMPLEX, .8,
                    text_color, 2, lineType=cv2.LINE_AA)

    else:
        for pt in pt_indices:
            cv2.circle(frame, points[pt], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, message, (30, 25), cv2.FONT_HERSHEY_SIMPLEX, .8,
                        text_color, 1, lineType=cv2.LINE_AA)






    cv2.imshow('Attentive Screen', frame)


