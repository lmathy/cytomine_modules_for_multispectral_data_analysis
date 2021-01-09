

for label in ["A11", "B4", "B7", "D13", "E3", "F9", "F12", "J4", "J6", "J11", "K3", "L2", "M2", "M3", "M4"]:
    string = "\\begin{figure}\n\centering\n\\begin{subfigure}\n{\includegraphics[width=.30\linewidth]{" + label + "_real_classes_with_quality_test.png}}\n\end{subfigure}%\n\\begin{subfigure}\n{\includegraphics[width=.30\linewidth]{" + label + "_predicted_classes.png}}\n\end{subfigure}\n\\begin{subfigure}\n\centering\n{\includegraphics[width=.30\linewidth]{" + label + "_accuracy_classes.png}}\n\end{subfigure}\n\caption{label core = " + label + ". Left image = real classes. Centre image = predicted classes. Right image = accuracy classes. For left and centre image: green = epithelium, red = stroma, purple = blood, blue = necrosis, yellow = pixel rejected by the quality test. For the right image: green = pixel correctly classified, red = pixel wrongly classified}\n\label{fig:" + label + "_images}\n\end{figure}\n\n"
    print(string)

