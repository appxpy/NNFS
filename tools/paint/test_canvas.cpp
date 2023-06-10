#include "test_canvas.h"
#include <iostream>

void TestCanvas::testClearImage_positive()
{
    Canvas canvas;
    canvas.clearImage();

    QImage image = canvas.pixmap().toImage();
    QColor pixelColor = image.pixelColor(0, 0);

    QVERIFY(pixelColor == Qt::black);
}

void TestCanvas::testClearImage_negative()
{
    Canvas canvas;
    canvas.clearImage();

    QImage image = canvas.pixmap().toImage();
    QColor pixelColor = image.pixelColor(0, 0);

    QVERIFY(pixelColor != Qt::white);
}

void TestCanvas::testRestartCanvas_positive()
{
    Canvas canvas;
    canvas.clearImage();
    canvas.restartCanvas();

    QImage image = canvas.pixmap().toImage();
    QColor pixelColor = image.pixelColor(0, 0);

    QVERIFY(pixelColor == Qt::black);
}

void TestCanvas::testRestartCanvas_negative()
{
    Canvas canvas;
    canvas.clearImage();
    canvas.restartCanvas();

    QImage image = canvas.pixmap().toImage();
    QColor pixelColor = image.pixelColor(0, 0);

    QVERIFY(pixelColor != Qt::white);
}

void TestCanvas::testGetMatrix_positive()
{
    Canvas canvas;
    Eigen::MatrixXd matrix = canvas.getMatrix();

    QCOMPARE(matrix.rows(), 1);
    QCOMPARE(matrix.cols(), 784);
}

void TestCanvas::testGetMatrix_negative()
{
    Canvas canvas;
    Eigen::MatrixXd matrix = canvas.getMatrix();

    QVERIFY(matrix.rows() != 0);
    QVERIFY(matrix.cols() != 0);
}

void TestCanvas::testMousePressEvent_positive()
{
    Canvas canvas;
    QTest::mousePress(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));

    QVERIFY(canvas.isDrawing());
}

void TestCanvas::testMousePressEvent_negative()
{
    Canvas canvas;

    QVERIFY(!canvas.isDrawing());
}

void TestCanvas::testMouseMoveEvent_positive()
{
    Canvas canvas;
    QTest::mousePress(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));

    QTest::mouseMove(&canvas, QPoint(10, 10));

    QImage image = canvas.pixmap().toImage();
    int pixelColor = (int)qGray(image.pixel(10, 10));

    QVERIFY(pixelColor > 128);
}

void TestCanvas::testMouseMoveEvent_negative()
{
    Canvas canvas;
    QTest::mousePress(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));

    QTest::mouseMove(&canvas, QPoint(10, 10));

    QImage image = canvas.pixmap().toImage();

    int pixelColor = (int)qGray(image.pixel(100, 100));
    QVERIFY(pixelColor == 0);
}

void TestCanvas::testMouseReleaseEvent_positive()
{
    Canvas canvas;
    QTest::mousePress(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));
    QTest::mouseMove(&canvas, QPoint(10, 10));
    QTest::mouseRelease(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));

    Eigen::MatrixXd matrix = canvas.getMatrix();

    QVERIFY(matrix.sum() != 0.0);
}

void TestCanvas::testMouseReleaseEvent_negative()
{
    Canvas canvas;
    QTest::mousePress(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));

    QTest::mouseRelease(&canvas, Qt::LeftButton, Qt::NoModifier, QPoint(10, 10));

    Eigen::MatrixXd matrix = canvas.getMatrix();

    QVERIFY(matrix.sum() == 0.0);
}

QTEST_MAIN(TestCanvas)
