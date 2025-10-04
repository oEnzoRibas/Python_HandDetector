# Relatório de Uso do Algoritmo de Detecção

## Introdução

Este relatório apresenta a aplicação de um algoritmo de detecção de mãos desenvolvido em Java utilizando a biblioteca OpenCV, no contexto do projeto de Robótica.

## Objetivo

O objetivo principal foi identificar e rastrear mãos em tempo real a partir de imagens capturadas pela webcam, permitindo a interação com o braço robótico por meio de gestos.

## Metodologia

- Utilização do OpenCV para captura de vídeo, pré-processamento (espelhamento, suavização, conversão de cor e segmentação por tons de pele).
- Aplicação de operações morfológicas e remoção de ruídos para melhorar a segmentação.
- Detecção de contornos e filtragem por área para identificar a mão.
- Cálculo do contorno convexo (convex hull) e defeitos de convexidade para estimar o número de dedos visíveis.
- Classificação do gesto com base na contagem de dedos detectados.

## Resultados

- O algoritmo funcionou de forma estável em ambientes bem iluminados, detectando e classificando gestos simples (como punho fechado e contagem de dedos).
- Em cenários com fundo complexo ou iluminação inadequada, a precisão da detecção foi reduzida.
- O tempo de processamento foi suficiente para aplicações em tempo real, com resposta visual imediata na interface.


## Performance do Algoritmo

O algoritmo de detecção apresentou desempenho satisfatório nos testes realizados, atendendo aos resultados esperados para a maioria dos cenários avaliados. A taxa de acerto foi consistente, com detecção eficiente das mãos em diferentes condições de iluminação e posicionamento. Pequenas variações de precisão foram observadas em situações de oclusão parcial, mas, de modo geral, o desempenho está alinhado com os objetivos do projeto.

## Reconhecimento de Mãos e Gestos

Além do reconhecimento eficiente das mãos, o algoritmo demonstrou capacidade satisfatória na identificação de gestos, ampliando as possibilidades de interação com o sistema. O reconhecimento de gestos foi testado em diferentes cenários, apresentando resultados promissores para comandos básicos.

## Análise de Dados

Para aprofundar a avaliação do desempenho, serão coletados e analisados dados quantitativos utilizando scripts em Python. Esses dados permitirão uma análise estatística detalhada dos resultados, identificando padrões e possíveis pontos de melhoria no algoritmo.

## Conclusão

A implementação do algoritmo de detecção de mãos demonstrou ser eficiente para o controle do braço robótico via gestos, embora apresente limitações em condições adversas de iluminação e fundo.

## Próximos Passos

- Aprimorar o pré-processamento das imagens para maior robustez em diferentes condições.
- Explorar outros classificadores e técnicas de aprendizado de máquina para melhorar a precisão.
- Integrar feedback visual mais detalhado para o usuário durante a operação.
- Realizar testes adicionais com uma variedade maior de gestos e condições ambientais.
- Coletar e analisar dados quantitativos para otimização contínua do algoritmo.
- Analisar dados coletados para identificar áreas de melhoria e ajustar o algoritmo conforme necessário.
- Gerar gráficos e relatórios detalhados para documentar o desempenho do sistema.


## Versão Atual do Código

```java
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.util.ArrayList;
import java.util.List;

public class HandDetector {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        VideoCapture camera = new VideoCapture(0);
        if (!camera.isOpened()) {
            System.out.println("❌ Erro ao abrir a webcam.");
            return;
        }

        Mat frame = new Mat();
        Mat mask = new Mat();
        Mat hierarchy = new Mat();

        while (true) {
            if (!camera.read(frame) || frame.empty()) break;

            // Espelha a imagem (modo selfie)
            Core.flip(frame, frame, 1);

            // Suaviza ruídos
            Imgproc.GaussianBlur(frame, frame, new Size(5,5), 0);

            // Converte para YCrCb e aplica máscara para tons de pele
            Mat ycrcb = new Mat();
            Imgproc.cvtColor(frame, ycrcb, Imgproc.COLOR_BGR2YCrCb);
            Core.inRange(ycrcb, new Scalar(0,133,77), new Scalar(255,173,127), mask);

            // Limpeza morfológica
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel);
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel);

            // Remoção de pequenos ruídos
            Imgproc.medianBlur(mask, mask, 5);

            // Encontra contornos
            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(mask.clone(), contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

            double maxArea = 5000; // aumenta o filtro para ignorar pequenos contornos
            int index = -1;
            for (int i = 0; i < contours.size(); i++) {
                double area = Imgproc.contourArea(contours.get(i));
                if (area > maxArea) {
                    maxArea = area;
                    index = i;
                }
            }

            if (index != -1) {
                MatOfPoint contour = contours.get(index);

                // Aproximação poligonal
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                Imgproc.approxPolyDP(contour2f, contour2f, 3, true);
                MatOfPoint approxContour = new MatOfPoint();
                contour2f.convertTo(approxContour, CvType.CV_32S);

                Imgproc.drawContours(frame, contours, index, new Scalar(0,255,0), 2);

                // Convex Hull
                MatOfInt hull = new MatOfInt();
                Imgproc.convexHull(approxContour, hull);
                MatOfPoint hullPoints = hullPointsFromIndices(approxContour, hull);
                List<MatOfPoint> hullList = new ArrayList<>();
                hullList.add(hullPoints);
                Imgproc.drawContours(frame, hullList, 0, new Scalar(255,0,0), 2);

                // Convexity Defects
                MatOfInt4 defects = new MatOfInt4();
                Imgproc.convexityDefects(approxContour, hull, defects);

                int fingers = countFingers(defects, approxContour);
                String gesture = classifyGesture(fingers);

                Imgproc.putText(frame,
                        "Dedos: " + fingers + " - " + gesture,
                        new Point(20,40),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        1.0, new Scalar(0,255,0), 2);
            }

            // Mostra apenas uma janela com o resultado final
            HighGui.imshow("Detecção de Mão", frame);

            // Delay aumentado para melhorar estabilidade
            try { Thread.sleep(150); } catch (InterruptedException e) { e.printStackTrace(); }

            if (HighGui.waitKey(1) == 27) break; // ESC para sair
        }

        camera.release();
        HighGui.destroyAllWindows();
    }

    // ---------------- Funções auxiliares ----------------

    private static MatOfPoint hullPointsFromIndices(MatOfPoint contour, MatOfInt hull) {
        Point[] contourPts = contour.toArray();
        int[] hullIdx = hull.toArray();
        Point[] hullPts = new Point[hullIdx.length];
        for (int i = 0; i < hullIdx.length; i++) hullPts[i] = contourPts[hullIdx[i]];
        MatOfPoint mop = new MatOfPoint();
        mop.fromArray(hullPts);
        return mop;
    }

    private static int countFingers(MatOfInt4 defects, MatOfPoint contour) {
        if (defects.empty()) return 0;
        int[] arr = defects.toArray();
        Point[] points = contour.toArray();
        int count = 0;
        for (int i = 0; i < arr.length; i += 4) {
            int startIdx = arr[i];
            int endIdx = arr[i+1];
            int farIdx = arr[i+2];
            float depth = arr[i+3] / 256.0f;
            if (depth > 25) { // aumenta profundidade mínima para melhor precisão
                double angle = calcAngle(points[startIdx], points[farIdx], points[endIdx]);
                if (angle < 85) count++; // ângulo máximo ajustado
            }
        }
        return Math.min(5, count + 1);
    }

    private static double calcAngle(Point a, Point b, Point c) {
        double ab = dist(a,b);
        double bc = dist(b,c);
        double ac = dist(a,c);
        double angle = Math.acos((ab*ab + bc*bc - ac*ac)/(2*ab*bc));
        return Math.toDegrees(angle);
    }

    private static double dist(Point p1, Point p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return Math.sqrt(dx*dx + dy*dy);
    }

    private static String classifyGesture(int fingers) {
        switch (fingers) {
            case 0: return "Fist";
            case 1: return "1 Fingers";
            case 2: return "2 Fingers";
            case 3: return "3 F312213ingers";
            case 4: return "4 Fingers";
            case 5: return "58 Fingers";
            default: return "Nada encontrado";
        }
    }
}
```

