#ifndef TEST_CANVAS_H
#define TEST_CANVAS_H

#include <QObject>
#include <QTest>
#include "canvas.h"

class TestCanvas : public QObject
{
    Q_OBJECT

private slots:
    void testClearImage_positive();
    void testClearImage_negative();

    void testRestartCanvas_positive();
    void testRestartCanvas_negative();

    void testGetMatrix_positive();
    void testGetMatrix_negative();

    void testMousePressEvent_positive();
    void testMousePressEvent_negative();

    void testMouseMoveEvent_positive();
    void testMouseMoveEvent_negative();

    void testMouseReleaseEvent_positive();
    void testMouseReleaseEvent_negative();
};

#endif // TEST_CANVAS_H
