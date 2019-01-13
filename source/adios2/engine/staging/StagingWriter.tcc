/*
 * Distributed under the OSI-approved Apache License, Version 2.0.  See
 * accompanying file Copyright.txt for details.
 *
 * StagingWriter.tcc
 *
 *  Created on: Nov 1, 2018
 *      Author: Jason Wang
 */

#ifndef ADIOS2_ENGINE_STAGINGWRITER_TCC_
#define ADIOS2_ENGINE_STAGINGWRITER_TCC_

#include "StagingWriter.h"

#include <iostream>

namespace adios2
{
namespace core
{
namespace engine
{

template <class T>
void StagingWriter::PutSyncCommon(Variable<T> &variable, const T *data)
{
    Log(5, "Staging Writer " + std::to_string( m_MpiRank) + " PutSync(" + variable.m_Name + ") start. Current step " + std::to_string(m_CurrentStep));
    if (m_IsActive)
    {
        variable.SetData(data);
        m_DataManSerializer.PutVar(variable, m_Name, CurrentStep(), m_MpiRank, m_FullAddresses[rand()%m_FullAddresses.size()], Params());
    }
    Log(5, "Staging Writer " + std::to_string( m_MpiRank) + " PutSync(" + variable.m_Name + ") end. Current step " + std::to_string(m_CurrentStep));
}

template <class T>
void StagingWriter::PutDeferredCommon(Variable<T> &variable, const T *data)
{
    Log(5, "Staging Writer " + std::to_string( m_MpiRank) + " PutDeferred(" + variable.m_Name + ") start. Current step " + std::to_string(m_CurrentStep));
    if (m_IsActive)
    {
        variable.SetData(data);
        m_DataManSerializer.PutVar(variable, m_Name, CurrentStep(), m_MpiRank, m_FullAddresses[rand()%m_FullAddresses.size()], Params());
    }
    Log(5, "Staging Writer " + std::to_string( m_MpiRank) + " PutDeferred(" + variable.m_Name + ") end. Current step " + std::to_string(m_CurrentStep));
}

} // end namespace engine
} // end namespace core
} // end namespace adios2

#endif // ADIOS2_ENGINE_STAGINGWRITER_TCC_
